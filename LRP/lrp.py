import torch
import copy
import pickle
import numpy as np
from . import rules
from . import utils
class LRP():
    '''
    torch implementation of LRP
    http://www.heatmapping.org/tutorial/
    :param rule:str: name of used rule
    '''

    def __init__(self, model, rule):
        self.model = copy.deepcopy(model)
        self.model = self.model.eval()
        # put hook to each basic layer
        self.hooks = [Hook(num, module) for num, module
                      in enumerate(utils.flatten_model(self.model))]
        self.output = None
        self.rule = rules.__getattr__(rule) #rule - str name of rule

    def forward(self, input_):
        self.output = self.model(input_)
        return self.output

    def relprop(self, R):
        global MODEL_DEPTH
        MODEL_DEPTH = len(self.hooks)
        assert self.output is not None, 'Forward pass not performed. Do .forward() first'
        for h in self.hooks[::-1]:
            R = h.relprop(R, self.rule)
        return R

    __call__ = relprop


class Hook(object):
    '''
    Hook will be add to each component of NN
    DOTO get ordered according to forwardpass layer list https://discuss.pytorch.org/t/list-layers-nodes-executed-in-order-of-forward-pass/46610
    '''

    def __init__(self, num, module):
        self.num = num  # serial number of basic layer
        self.module = module
        self.hookF = self.module.register_forward_hook(self.hook_fnF)

    #         self.hookB = module.register_backward_hook(self.hook_fnB)

    def hook_fnF(self, module, input_, output):
        assert len(input_) == 1, 'batch size should be eaual to 1'
        self.input = copy.deepcopy(input_[0].clone().detach())
        self.input.requires_grad_(True)
        self.output = output

    #     def hook_fnB(self, module, input, output):
    #         self.input = input
    #         self.output = output

    def close(self):
        self.hookF.remove()

    def relprop(self, R, rule):
        '''
        Layer type specific propogation of relevance
        '''
        R = R.view(self.output.shape)  # stitching ".view" step, frequantly in classifier
        layer_name = self.module._get_name()
        Ri = globals()[layer_name].relprop(utils.copy_module(self.module),
                                           self.input, R, self.num, rule)
        Ri = Ri.clone().detach()
        if self.input.grad is not None:
            self.input.grad.zero_()
        return Ri

    def __del__(self):
        self.close()
        del_super = getattr(super(Hook, self), "__del__", None)
        if callable(del_super):
            del_super()


# Initial idea was to overload backward pass, 
# but then I stick with hooks. Here for each layer I define 
# relprop method


class ReLU(torch.nn.ReLU):

    @staticmethod
    def relprop(module, input_, R, num, rule):
        return R


class Dropout(torch.nn.Dropout):

    @staticmethod
    def relprop(module, input_, R, num, rule):
        return R


class Linear(torch.nn.Linear):

    @staticmethod
    def relprop(module, input_, R, num, rule):
        if num == MODEL_DEPTH - 1: #if last layer
            R = rule(module, input_, R)
        elif num == 0: #if first layer
            R = rules.z_box_rule(module, input_, R, lowest=-1,
              highest=1) #image should be maped to [-1, 1] intv
        else:
            R = rule(module, input_, R)
        return R


class Conv2d(torch.nn.Conv2d):

    @staticmethod
    def relprop(module, input_, R, num, rule):
        if num == 0:
            R = rules.z_box_rule(module, input_, R, lowest=-1,
                    highest=1)
        else: #nextconvolitional layer
            R = rule(module, input_, R)
        return R


class MaxPool2d(torch.nn.MaxPool2d):

    @staticmethod
    def relprop(module, input_, R, num, rule):
        R = rules.z_rule(module, input_, R)
        return R
