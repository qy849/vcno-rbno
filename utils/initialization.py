import torch.nn.init as init

def get_initializer(name):

    if name == 'xavier_uniform':
        return init.xavier_uniform_
    elif name == 'xvaier_normal':
        return init.xavier_normal_
    elif name == 'kaiming_uniform':
        return init.kaiming_uniform_
    elif name == 'kaiming_normal':
        return init.kaiming_normal_
    elif name == 'uniform':
        return init.uniform_
    elif name == 'normal':
        return init.normal_
    elif name == 'trunc_normal':
        return init.trunc_normal_
    else:
        raise NotImplementedError