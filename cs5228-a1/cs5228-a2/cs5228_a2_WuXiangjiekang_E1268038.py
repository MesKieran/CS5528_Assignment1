import numpy as np

from sklearn.metrics.pairwise import euclidean_distances



def get_noise_dbscan(X, eps=0.0, min_samples=0):
    
    core_point_indices, noise_point_indices = None, None
    
    #########################################################################################
    ### Your code starts here ###############################################################
    
    ### 2.1 a) Identify the indices of all core points
    
    
    ### Your code ends here #################################################################
    #########################################################################################
    
    
    distances = euclidean_distances(X, X)
    num_neighbors = np.sum(distances <= eps, axis=1)
    core_point_indices = np.where(num_neighbors >= min_samples)[0]
    
    # #########################################################################################
    # ### Your code starts here ###############################################################
    
    # ### 2.1 b) Identify the indices of all noise points ==> noise_point_indices 
    # border_point_indices = []
    # for i in range(X.shape[0]):
    #     if i not in core_point_indices and np.any(distances[i, core_point_indices] <= eps):
    #         border_point_indices.append(i)
            
    # all_indices = np.arange(X.shape[0])
    # noise_point_indices = np.setdiff1d(all_indices, np.concatenate((core_point_indices, border_point_indices)))
    core_point_mask = np.zeros(X.shape[0], dtype=bool)
    core_point_mask[core_point_indices] = True
    is_border_point = np.logical_and(~core_point_mask, np.any(distances[:, core_point_indices] <= eps, axis=1))
    border_point_indices = np.where(is_border_point)[0]
    all_indices = np.arange(X.shape[0])
    noise_point_indices = np.setdiff1d(all_indices, np.concatenate((core_point_indices, border_point_indices)))
    
    ### Your code ends here #################################################################
    #########################################################################################
    
    return core_point_indices, noise_point_indices

