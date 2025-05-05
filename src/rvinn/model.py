import torch
import torch.nn.functional
import numpy as np
import matplotlib.pyplot as plt

from . import modules # rvinn modules.py

class Model():
    def __init__(self, device, **kwargs):
        # data and collocation points
        self.device = device
        self.t_train = torch.tensor(kwargs.get('t_train'), requires_grad=True).float().to(self.device)
        self.t_f = torch.tensor(kwargs.get('t_f'), requires_grad=True).float().to(self.device)
        self.spliced_train = torch.tensor(kwargs.get('spliced_train'), requires_grad=False).float().to(self.device)
        self.unspliced_train = torch.tensor(kwargs.get('unspliced_train'), requires_grad=False).float().to(self.device)

        # neural networks from rvinn modules.py
        self.config = kwargs.get('config', None)
        if self.config:
            # load user settings
            self.NN = modules.NN(self.config["layers"]["gene-expression_module"]).to(self.device)
            self.NN_k1 = modules.NN(self.config["layers"]["transcription_module"]).to(self.device)
            self.NN_k23 = modules.NN(self.config["layers"]["post-transcription_module"]).to(self.device)
            self.Adam_lr = self.config["opt_settings"]["Adam_lr"]
            self.Adam_max = self.config["opt_settings"]["Adam_max"]
            self.LBFGS_lr = self.config["opt_settings"]["LBFGS_lr"]
            self.LBFGS_max = self.config["opt_settings"]["LBFGS_max"]
            self.SA_ODE_init = self.config["opt_settings"]["SA_ODE_init"]
            self.SA_AUX_init = self.config["opt_settings"]["SA_AUX_init"]
        else:
            # default settings
            self.config = {"layers": {
                        "gene-expression_module": [
                            1,
                            2*(len(self.t_train)),
                            2*(len(self.t_train)),
                            2
                        ],
                        "transcription_module": [
                            1,
                            max(32, 2*len(self.t_train)),
                            max(32, 2*len(self.t_train)),
                            1
                        ],
                        "post-transcription_module": [
                            1,
                            max(32, 2*len(self.t_train)),
                            max(32, 2*len(self.t_train)),
                            2
                        ]},
                        "opt_settings": {
                        "Adam_lr": 0.001,
                        "Adam_max": 3000,
                        "LBFGS_lr": 1.0,
                        "LBFGS_max": 5000,
                        "SA_ODE_init": 1/len(self.t_train),
                        "SA_AUX_init": 0.01
                        }}
            self.NN = modules.NN(self.config["layers"]["gene-expression_module"]).to(self.device)
            self.NN_k1 = modules.NN(self.config["layers"]["transcription_module"]).to(self.device)
            self.NN_k23 = modules.NN(self.config["layers"]["post-transcription_module"]).to(self.device)
            self.Adam_lr = self.config["opt_settings"]["Adam_lr"]
            self.Adam_max = self.config["opt_settings"]["Adam_max"]
            self.LBFGS_lr = self.config["opt_settings"]["LBFGS_lr"]
            self.LBFGS_max = self.config["opt_settings"]["LBFGS_max"]
            self.SA_ODE_init = self.config["opt_settings"]["SA_ODE_init"]
            self.SA_AUX_init = self.config["opt_settings"]["SA_AUX_init"]

        # print config option
        self.print_config = kwargs.get('print_config', False)

        # user assumptions
        self.init_steady_loss = kwargs.get('init_steady', False)
        self.init_data_loss = kwargs.get('init_data', False)

        # normalized real data or not
        # default is normalized real data
        self.normalization = kwargs.get('normalization', True)

        # print loss verbose option
        self.verbose = kwargs.get('loss_verbose', True)
        
        # self-adaptive weight
        self.SA_ODE = modules.SelfAdaptiveWeight(self.SA_ODE_init).to(self.device) # must be float
        self.SA_AUX = modules.SelfAdaptiveWeight(self.SA_AUX_init).to(self.device)


        # merge three NN's parameters
        params = list(self.NN.parameters()) + list(self.NN_k1.parameters()) + list(self.NN_k23.parameters())
        params_SA_ODE = list(self.SA_ODE.parameters())
        params_SA_AUX = list(self.SA_AUX.parameters())

        params_SA = params_SA_ODE + params_SA_AUX

        total_params = params + params_SA 

        # history
        self.history = {
            'loss': [],
            'loss_DATA': [],
            'loss_ODE': [],
            'loss_AUXILIARY': [],
            'lambda_ODE': [],
            'lambda_AUX': [],
            }
        
        # optimizer
        self.optimizer_Adam = torch.optim.Adam(params=total_params)
        self.iter = 0

        self.optimizer_Total = torch.optim.LBFGS(
            total_params,
            lr = self.LBFGS_lr,
            max_iter = self.LBFGS_max,
            max_eval = self.LBFGS_max,
            history_size = 10,
            tolerance_grad = 1e-5,
            tolerance_change = 1.0 * np.finfo(float).eps,
            line_search_fn = "strong_wolfe"   
        )

        self.optimizer_SA_ODE = torch.optim.Adam(params = params_SA_ODE, lr = self.Adam_lr)
        self.optimizer_SA_AUX = torch.optim.Adam(params = params_SA_AUX, lr = self.Adam_lr)

        if self.print_config:
            print("Model configuration:")
            print(self.config["layers"])
            print(self.config["opt_settings"])

            print(f"init_steady is set to: {self.init_steady_loss}")
            print(f"init_data is set to: {self.init_data_loss}")
        else:
            pass

    def net_u(self, t):
        STATE = self.NN(t)
        Sp = STATE[:,0:1] # Tensor = [Timepoints, 1]
        Un = STATE[:,1:2] # Tensor = [Timepoints, 1]
        if self.normalization:
            Sp = torch.nn.functional.sigmoid(Sp) # should be positive and capped at 1 
            Un = torch.nn.functional.sigmoid(Un) # should be positive and capped at 1 
        else:
            Sp = torch.nn.functional.softplus(Sp) # should be positive
            Un = torch.nn.functional.softplus(Un) # should be positive
        return Sp, Un

    def net_ks(self,t):
        k1 = self.NN_k1(t)
        ks = self.NN_k23(t)
        k2 = ks[:,0:1] # Tensor = [Timepoints, 1]
        k3 = ks[:,1:2] # Tensor = [Timepoints, 1]
        k1 = torch.nn.functional.softplus(k1) # should be positive
        k2 = torch.nn.functional.softplus(k2) # should be positive
        k3 = torch.nn.functional.softplus(k3) # should be positive
        return k1, k2, k3 # net_ks output

    def net_f(self, t):
        #load time-varying parameters
        k1, k2, k3 = self.net_ks(t)

        #predicting time series of Sp, Un from net_u
        Sp, Un = self.net_u(t)

        #Time derivatives
        Sp_t = torch.autograd.grad(Sp, t,
                                  grad_outputs=torch.ones_like(Sp),
                                  retain_graph=True,
                                  create_graph=True
                                  )[0]
        Un_t = torch.autograd.grad(Un, t,
                                  grad_outputs=torch.ones_like(Un),
                                  retain_graph=True,
                                  create_graph=True
                                  )[0]
        k1_t = torch.autograd.grad(k1, t,
                                  grad_outputs=torch.ones_like(k1),
                                  retain_graph=True,
                                  create_graph=True
                                  )[0]
        k2_t = torch.autograd.grad(k2, t,
                                  grad_outputs=torch.ones_like(k2),
                                  retain_graph=True,
                                  create_graph=True
                                  )[0]
        k3_t = torch.autograd.grad(k3, t,
                                  grad_outputs=torch.ones_like(k3),
                                  retain_graph=True,
                                  create_graph=True
                                  )[0]

        #Residuals (ODE)
        f_Sp = Sp_t - (k2*Un - k3*Sp)
        f_Un = Un_t - (k1 - k2*Un)

        return f_Sp, f_Un, k1, k2, k3, Sp_t, Un_t, k1_t, k2_t, k3_t 
    


    

    def loss_func(self):
        Sp_pred, Un_pred = self.net_u(self.t_train)
        
        Sp_f, Un_f, k1, k2, k3, Sp_t, Un_t, k1_t, k2_t, k3_t = self.net_f(self.t_f)

        sa_ODE = self.SA_ODE()
        sa_AUX = self.SA_AUX()

        loss_DATA = torch.mean((self.spliced_train - Sp_pred)**2) + torch.mean((self.unspliced_train - Un_pred)**2)
        if self.init_data_loss:
            loss_DATA += torch.mean((Sp_pred[0] - self.spliced_train[0])**2) + torch.mean((Un_pred[0] - self.unspliced_train[0])**2)

        loss_ODE = sa_ODE*torch.mean(Sp_f**2) + sa_ODE*torch.mean(Un_f**2)
        if self.init_steady_loss:
            loss_ODE += sa_ODE*torch.mean((k1[0] - k2[0]*Un_pred[0])**2) + sa_ODE*torch.mean((k2[0]*Un_pred[0] - k3[0]*Sp_pred[0])**2)
            
        loss_AUXILIARY = sa_AUX*torch.mean(torch.abs(k2_t))

        lambda_ODE = torch.min(sa_ODE)
        lambda_AUX = torch.min(sa_AUX)
        loss = loss_DATA + loss_ODE + loss_AUXILIARY

        self.optimizer_SA_ODE.zero_grad() 
        self.optimizer_SA_AUX.zero_grad()
        self.optimizer_Total.zero_grad() 
        loss.backward()

        self.history['loss'].append(loss.item())
        self.history['loss_DATA'].append(loss_DATA.item())
        self.history['loss_ODE'].append(loss_ODE.item()/lambda_ODE.item())
        self.history['loss_AUXILIARY'].append(loss_AUXILIARY.item()/lambda_AUX.item())
        self.history['lambda_ODE'].append(lambda_ODE.item())
        self.history['lambda_AUX'].append(lambda_AUX.item())

        self.iter += 1
        if (self.iter % 100 == 0) and self.verbose:
              print(
                 'Loss: %.5f, Loss_DATA: %.5f, Loss_ODE: %.5f, Loss_AUX: %.5f, lambda_ODE: %.5f, lambda_AUX: %.5f, LBFGS_Itr: %d' %
                 (
                     loss.item(),
                     loss_DATA.item(),
                     loss_ODE.item()/lambda_ODE.item(),
                     loss_AUXILIARY.item()/lambda_AUX.item(),
                     lambda_ODE.item(),
                     lambda_AUX.item(),
                     self.iter
                 )
                  )
        return loss

    def train(self):
        nIter = self.Adam_max
        for epoch in range(nIter):
            Sp_pred, Un_pred = self.net_u(self.t_train)
            
            Sp_f, Un_f, k1, k2, k3, Sp_t, Un_t, k1_t, k2_t, k3_t = self.net_f(self.t_f)

            sa_ODE = self.SA_ODE()
            sa_AUX = self.SA_AUX()

            loss_DATA = torch.mean((self.spliced_train - Sp_pred)**2) + torch.mean((self.unspliced_train - Un_pred)**2)
            if self.init_data_loss:
                loss_DATA += torch.mean((Sp_pred[0] - self.spliced_train[0])**2) + torch.mean((Un_pred[0] - self.unspliced_train[0])**2)
            
            loss_ODE = sa_ODE*torch.mean(Sp_f**2) + sa_ODE*torch.mean(Un_f**2)
            if self.init_steady_loss:
                loss_ODE += sa_ODE*torch.mean((k1[0] - k2[0]*Un_pred[0])**2) + sa_ODE*torch.mean((k2[0]*Un_pred[0] - k3[0]*Sp_pred[0])**2)
            
            loss_AUXILIARY = sa_AUX*torch.mean(torch.abs(k2_t))


            lambda_ODE = torch.min(sa_ODE)
            lambda_AUX = torch.min(sa_AUX)
            loss = loss_DATA + loss_ODE + loss_AUXILIARY

            # history
            self.history['loss'].append(loss.item())
            self.history['loss_DATA'].append(loss_DATA.item())
            self.history['loss_ODE'].append(loss_ODE.item()/lambda_ODE.item())
            self.history['loss_AUXILIARY'].append(loss_AUXILIARY.item()/lambda_AUX.item())
            self.history['lambda_ODE'].append(lambda_ODE.item())
            self.history['lambda_AUX'].append(lambda_AUX.item())

            # Backward and optimize
            self.optimizer_SA_ODE.zero_grad() 
            self.optimizer_SA_AUX.zero_grad() 
            self.optimizer_Adam.zero_grad()

            loss.backward()
            self.optimizer_Adam.step()

            if (epoch % 100 == 0) and self.verbose:
                print(
                    'Loss: %.5f, Loss_DATA: %.5f, Loss_ODE: %.5f, Loss_AUX: %.5f, lambda_ODE: %.5f, weight_AUX: %.5f, Adam_Itr: %d' %
                        (
                        loss.item(),
                        loss_DATA.item(),
                        loss_ODE.item()/lambda_ODE.item(),
                        loss_AUXILIARY.item()/lambda_AUX.item(),
                        lambda_ODE.item(),
                        lambda_AUX.item(),
                        epoch
                        )
                        )
        # Backward and optimize
        self.optimizer_Total.step(self.loss_func) #Adam to LBFGS

    def predict(self, t):
        # prediction time window to torch.tensor
        t = torch.tensor(t, requires_grad=True).float().to(self.device)

        self.NN.eval()
        self.NN_k1.eval()
        self.NN_k23.eval()
        Sp, Un = self.net_u(t)
        k1, k2, k3 = self.net_ks(t)
        Sp_f, Un_f, k1, k2, k3, Sp_t, Un_t, k1_t, k2_t, k3_t = self.net_f(t)
        Sp = Sp.detach().cpu().numpy()
        Un = Un.detach().cpu().numpy()
        k1 = k1.detach().cpu().numpy()
        k2 = k2.detach().cpu().numpy()
        k3 = k3.detach().cpu().numpy()
        Sp_t = Sp_t.detach().cpu().numpy()
        Un_t = Un_t.detach().cpu().numpy()
        k1_t = k1_t.detach().cpu().numpy()
        k2_t = k2_t.detach().cpu().numpy()
        k3_t = k3_t.detach().cpu().numpy()
        return Sp, Un, k1, k2, k3, Sp_t, Un_t, k1_t, k2_t, k3_t

    def history_plot(self):
        fig = plt.figure(figsize = (16,12))
        ax_loss = fig.add_subplot(2, 1, 1)
        ax_SA = fig.add_subplot(2, 1, 2)

        ax_loss.plot(self.history['loss'], label = "Total_loss")
        ax_loss.plot(self.history['loss_DATA'], label = "loss_DATA")
        ax_loss.plot(self.history['loss_ODE'], label = "loss_ODE")
        ax_loss.plot(self.history['loss_AUXILIARY'], label = "loss_AUXILIARY")
        ax_loss.set_yscale("log")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_xlabel("Iteration")
        ax_loss.legend()

        ax_SA.plot(self.history['lambda_ODE'], label = "lambda_ODE")
        ax_SA.plot(self.history['lambda_AUX'], label = "lambda_AUX")
        ax_SA.set_xlabel("Iteration")
        ax_SA.set_ylabel("Weight value")
        ax_SA.legend()
        #ax_loss.set_yscale("log")
