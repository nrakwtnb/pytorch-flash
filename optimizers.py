

def get_optimzier(optimizer_info, model):
    opt_name = optimizer_info['name']
    opt_args = optimizer_info['args']
    if opt_name in ['SGD']:
        from torch.optim import SGD
        return SGD(model.parameters(), **opt_args)
    elif opt_name in ['Adam']:
        from torch.optim import Adam
        return Adam(model.parameters(), **opt_args)
    elif opt_name in ['RMSprop']:
        from torch.optim import RMSprop
        return RMSprop(model.parameters(), **opt_args)
    elif opt_name in ['Adadelta']:
        from torch.optim import Adadelta
        return Adadelta(model.parameters(), **opt_args)
    elif opt_name in ['Adagrad']:
        from torch.optim import Adagrad
        return Adagrad(model.parameters(), **opt_args)
    else:
        assert False, "Invalid optimizer name"

