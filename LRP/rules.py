import torch
import numpy as np
import copy

def z_rule(module, input_, R):
    '''
    '''
    pself = module
    Z = pself(input_)
    S = R /(Z + (Z==0).float()*np.finfo(np.float32).eps)
    Z.backward(S)
    C = input_.grad
    R = input_ * C
    return R

def z_plus_rule(module, input_, R, keep_bias=False):
    '''
    if input constrained to positive only (e.g. Relu)
    '''
    pself = module
    if hasattr(pself, 'bias'):
        if not keep_bias:
            pself.bias.data.zero_()
    if hasattr(pself, 'weight'):
        pself.weight.data.clamp_(0, float('inf'))
    R = z_rule(pself, input_, R)
    return R


def z_epsilon_rule(module, input_, R, keep_bias=True):
    '''
    '''
    pself = module
    if hasattr(pself, 'bias'):
        if not keep_bias:
            pself.bias.data.zero_()
   # if hasattr(pself, 'weight'):
   #     pself.weight.data.clamp_(0, float('inf'))
    Z = pself(input_)
    S = R / (Z + ((input_>=0).float()*2-1)*np.finfo(np.float32).eps)
    Z.backward(S)
    C = input_.grad
    R = input_ * C
    return R


def alfa_beta_rule(module, input_, R, alfa=2, keep_bias=False):
    '''
    General rule, alfa = 1 is z_plus_rule case. 
    '''
    nself = copy.deepcopy(module)
    nself.weight.data.clamp_(-float('inf'), -np.finfo(np.float32).eps)
    pself = copy.deepcopy(module)
    pself.weight.data.clamp_(np.finfo(np.float32).eps, float('inf'))
    if hasattr(pself, 'bias'):
        if not keep_bias:
            pself.bias.data.zero_()
            nself.bias.data.zero_()
    inputA_ = copy.deepcopy(input_ + np.finfo(np.float32).eps)
    inputB_ = copy.deepcopy(input_ + np.finfo(np.float32).eps)
    inputA_.requires_grad_(True)
    inputA_.retain_grad()
    inputB_.requires_grad_(True)
    inputB_.retain_grad()
    ZA = pself(inputA_)
    SA = alfa*R/ZA
    ZA.backward(SA)
    ZB = nself(inputB_)
    SB = -1*(alfa-1)*R/ZB #TODO:why alfa-1 not alfa+beta = 1?
    ZB.backward(SB)
    Ri = input_*(inputA_.grad + inputB_.grad)
    return Ri

def z_box_rule(module, input_, R, lowest, highest, keep_bias=False):
    '''
    if input constrained to bounds lowest and highest
    '''
    assert input_.min() >= lowest
    assert input_.max() <= highest
    iself = copy.deepcopy(module)
    nself = copy.deepcopy(module)
    nself.weight.data.clamp_(-float('inf'), 0)
    pself = copy.deepcopy(module)
    pself.weight.data.clamp_(0, float('inf'))
    if hasattr(pself, 'bias'):
        if not keep_bias:
            pself.bias.data.zero_()
            nself.bias.data.zero_()
            iself.bias.data.zero_()
    L = torch.zeros_like(input_) + lowest
    H = torch.zeros_like(input_) + highest
    L.requires_grad_(True)
    L.retain_grad()
    H.requires_grad_(True)
    H.retain_grad()
    Z = iself(input_) - pself(L) - nself(H) + np.finfo(np.float32).eps
    S = R / (Z + (Z==0).float()*np.finfo(np.float32).eps)
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
    Z = pself(input_)
    S = R / pself(torch.ones_like(input_))
    Z.backward(S)
    R = input_.grad
    return R
