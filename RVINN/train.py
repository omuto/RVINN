import numpy as np
import pandas as pd
import os

def load_target(gene: str, df_Sp: pd.DataFrame, df_Un: pd.DataFrame, n_replicates: int) -> tuple:
    assert type(n_replicates) == int, "n_replicates must be an integer."
    Sp_new_star = df_Sp.loc[gene].to_numpy(dtype=np.float64).reshape(n_replicates, -1).T
    Un_new_star = df_Un.loc[gene].to_numpy(dtype=np.float64).reshape(n_replicates, -1).T
    return Sp_new_star, Un_new_star

def load_observation_period(experimental_timepoints_list: list):
    assert type(experimental_timepoints_list) == list, "experimental_timepoints_list must be a list."
    """ The experimental_timepoints_list must be in minutes """
    t_star = np.array(experimental_timepoints_list, dtype=np.float64).reshape(-1,1)
    end_min = int(np.max(t_star))
    start_min = int(np.min(t_star))
    """ Collocation points be generated in every minute """
    t_f = np.linspace(start_min, end_min, (end_min - start_min) + 1).reshape(-1,1)
    """ Input values for NN are in hours """
    t_star /= 60
    t_f /= 60
    assert np.max(t_star) == np.max(t_f), "The endpoint of t_star and t_f must be equal."
    return t_star, t_f

def create_save_directory(gene: str, experiment_name = None) -> str:
    """Create a directory to store the results and return the path."""
    current_directory = os.getcwd()

    if experiment_name is not None:
        relative_path_results = f'{experiment_name}/Results-{gene}/'
    else:
        relative_path_results = f'Inference/Results-{gene}/'

    save_results_to = os.path.join(current_directory, relative_path_results)

    if not os.path.exists(save_results_to):
        os.makedirs(save_results_to)

    return save_results_to

def save_results(gene: str, save_results_to: str, Sp_e, Un_e, k1_e, k2_e, k3_e, Sp_t_e, Un_t_e, k1_t_e, k2_t_e, k3_t_e):

    #save_results_to = create_save_directory(gene, experiment_name = 'train')

    """save the results to a file"""
    np.save(f'{save_results_to}k1_{gene}.npy', k1_e)
    np.save(f'{save_results_to}k2_{gene}.npy', k2_e)
    np.save(f'{save_results_to}k3_{gene}.npy', k3_e)
    np.save(f'{save_results_to}Sp_{gene}.npy', Sp_e)
    np.save(f'{save_results_to}Un_{gene}.npy', Un_e)
    
    # Mean and standard deviation 
    np.save(f'{save_results_to}k1_mean_{gene}.npy', k1_e.mean(axis=1))
    np.save(f'{save_results_to}k2_mean_{gene}.npy', k2_e.mean(axis=1))
    np.save(f'{save_results_to}k3_mean_{gene}.npy', k3_e.mean(axis=1))
    np.save(f'{save_results_to}Sp_mean_{gene}.npy', Sp_e.mean(axis=1))
    np.save(f'{save_results_to}Un_mean_{gene}.npy', Un_e.mean(axis=1))

    np.save(f'{save_results_to}k1_std_{gene}.npy', k1_e.std(axis=1))
    np.save(f'{save_results_to}k2_std_{gene}.npy', k2_e.std(axis=1))
    np.save(f'{save_results_to}k3_std_{gene}.npy', k3_e.std(axis=1))
    np.save(f'{save_results_to}Sp_std_{gene}.npy', Sp_e.std(axis=1))
    np.save(f'{save_results_to}Un_std_{gene}.npy', Un_e.std(axis=1))

    # derivative results
    np.save(f'{save_results_to}k1_t_{gene}.npy', k1_t_e)
    np.save(f'{save_results_to}k2_t_{gene}.npy', k2_t_e)
    np.save(f'{save_results_to}k3_t_{gene}.npy', k3_t_e)
    np.save(f'{save_results_to}Sp_t_{gene}.npy', Sp_t_e)
    np.save(f'{save_results_to}Un_t_{gene}.npy', Un_t_e)

    # derivative mean
    np.save(f'{save_results_to}k1_t_mean_{gene}.npy', k1_t_e.mean(axis=1))
    np.save(f'{save_results_to}k2_t_mean_{gene}.npy', k2_t_e.mean(axis=1))
    np.save(f'{save_results_to}k3_t_mean_{gene}.npy', k3_t_e.mean(axis=1))
    np.save(f'{save_results_to}Sp_t_mean_{gene}.npy', Sp_t_e.mean(axis=1))
    np.save(f'{save_results_to}Un_t_mean_{gene}.npy', Un_t_e.mean(axis=1))
