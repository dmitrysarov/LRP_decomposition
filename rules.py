import torch
import numpy as np
import copy

def z_rule(module, input_, R):
    '''
    '''
    pself = module
    Z = pself(input_) + np.finfo(np.float32).eps
    S = R / Z
    Z.backward(S)
    C = input_.grad
    R = input_ * C
    return R

def z_plus_rule(module, input_, R):
    '''
    if input constrained to positive only (e.g. Relu)
    '''
    pself = module
    if hasattr(pself, 'bias'):
        pself.bias.data.zero_()
    if hasattr(pself, 'weight'):
        pself.weight.data.clamp_(0, float('inf'))
    R = z_rule(pself, input_, R)
    return R


def z_epsilon_rule(module, input_, R):
    '''
    if input constrained to positive only (e.g. Relu)
    '''
    pself = module
    if hasattr(pself, 'bias'):
        pself.bias.data.zero_()
    if hasattr(pself, 'weight'):
        pself.weight.data.clamp_(0, float('inf'))
    Z = pself(input_) + np.finfo(np.float32).eps
    S = R / Z
    Z.backward(S)
    C = input_.grad
    R = input_ * C
    return R


def z_box_rule(module, input_, R, lowest, highest):
    '''
    if input constrained to bounds lowest and highest
    '''
    assert input_.min() >= lowest
    assert input_.max() <= highest
    iself = copy.deepcopy(module)
    iself.bias.data.zero_()
    nself = copy.deepcopy(module)
    nself.bias.data.zero_()
    nself.weight.data.clamp_(-float('inf'), 0)
    pself = copy.deepcopy(module)
    pself.bias.data.zero_()
    pself.weight.data.clamp_(0, float('inf'))
    L = torch.zeros_like(input_) + lowest
    H = torch.zeros_like(input_) + highest
    L.requires_grad_(True)
    L.retain_grad()
    H.requires_grad_(True)
    H.retain_grad()
    Z = iself(input_) - pself(L) - nself(H) + np.finfo(np.float32).eps
    S = R / Z
    Z.backward(S)
    R = input_ * input_.grad - L * L.grad - H * H.grad
    return R


def w2_rule(module, input_, R):
    '''
    if input is unconstrained
    '''
    pself = module
    if hasattr(pself, 'bias'):
        pself.bias.data.zero_()
    if hasattr(pself, 'weight'):
        pself.weight.data = pself.weight.data ** 2
    Z = pself(input_) + np.finfo(np.float32).eps
    S = R / pself(torch.ones_like(input_))
    Z.backward(S)
    R = input_.grad
    return R
