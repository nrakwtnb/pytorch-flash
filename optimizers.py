

def get_optimzier(optimizer_info, model):
    opt_name = optimizer_info['name']
    opt_info = optimizer_info['info']
    if opt_name in ['SGD']:
        from torch.optim import SGD
        return SGD(model.parameters(), **opt_info)
    elif opt_name in ['Adam']:
        from torch.optim import Adam
        return Adam(model.parameters(), **opt_info)
    else:
        assert False, "Invalid optimizer name"

