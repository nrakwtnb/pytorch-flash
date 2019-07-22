
def get_sampled_loader(data_loader, num_samples, attributions=['data' 'target'], seed=0):
    dataset = data_loader.dataset
    num_data = len(dataset)
    assert all([hasattr(dataset, a) for a in attributions])
    assert all([ len(dataset.__getattribute__(a)) == num_data for a in attributions])
    assert num_samples <= num_data
    
    import copy
    sampled_data_loader = copy.deepcopy(data_loader)
    
    import numpy as np
    np.random.seed(seed)
    sampled = np.random.randint(0,num_data,num_samples)
    for attr in attributions:
        # change the type accorind to the attribution type
        sampled_data_loader.dataset.__setattr__(attr, [dataset.__getattribute__(attr)[idx] for idx in sampled])
    return sampled_data_loader


