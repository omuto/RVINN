import torch
from collections import OrderedDict

# neural networks
class NN(torch.nn.Module):
    def __init__(self, layers):
        super(NN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = torch.nn.Tanh #activation function
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        def init_weights(m):
          if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight) 
            if m.bias is not None:
               torch.nn.init.normal_(m.bias) 

        # deploy layers 
        self.layers = torch.nn.Sequential(layerDict)
        #Xavier_initialization
        self.layers.apply(init_weights)

    def forward(self, t):
        out = self.layers(t)
        return out

class SelfAdaptiveWeight(torch.nn.Module):
    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg()

    @classmethod
    def grad_reverse(cls, x):
        return cls.GradReverse.apply(x)

    def __init__(self, init_value):
        """
        Parameters
        ----------
        init_value : float
            Initial value for the initial adaptive weight.
        """
        super().__init__()
        self.init_value = init_value
        self.weight = torch.nn.Parameter(self.init_value * torch.ones(1))

    def forward(self):
        # Apply gradient reversal to the weights for gradient ascent
        weight = self.grad_reverse(self.weight)
        return weight