
def get_sampled_loader(data_loader, num_samples, seed=0):
    num_data = len(data_loader.dataset.data)
    assert num_data == len(data_loader.dataset.targets) and num_samples <= num_data
    
    import copy
    sampled_data_loader = copy.copy(data_loader)
    
    import numpy as np
    np.random.seed(seed)
    sampled = np.random.randint(0,num_data,num_samples)
    sampled_data_loader.data = sampled_data_loader.dataset.data[sampled]
    sampled_data_loader.targets = sampled_data_loader.dataset.targets[sampled]
    return sampled_data_loader


