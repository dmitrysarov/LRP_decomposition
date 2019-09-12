import torch
import copy
import pickle
import numpy as np
import rules

def flatten_model(module):
    '''
    flatten modul to base operation like Conv2, Linear, ...
    '''
    modules_list = []
    for m_1 in module.children():

        if len(list(m_1.children())) == 0:
            modules_list.append(m_1)
        else:
            modules_list = modules_list + flatten_model(m_1)
    return modules_list


def copy_module(module):
    '''
    sometimes copy.deepcopy() does not work
    '''
    module = copy.deepcopy(pickle.loads(pickle.dumps(module)))
    module._forward_hooks.popitem()  # remove hooks from module copy
    return module


class LRP():
    '''
    torch implementation of LRP
    http://www.heatmapping.org/tutorial/
    '''

    def __init__(self, model):
        self.model = copy.deepcopy(model)
        self.model = self.model.eval()
        # put hook to each basic layer
        self.hooks = [Hook(num, module) for num, module
                      in enumerate(flatten_model(self.model))]
        self.output = None

    def forward(self, input_):
        self.output = self.model(input_)
        return self.output

    def relprop(self, R):
        global MODEL_DEPTH
        MODEL_DEPTH = len(self.hooks)
        assert self.output is not None, 'Forward pass not performed. Do .forward() first'
        for h in self.hooks[::-1]:
            R = h.relprop(R)
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

    def relprop(self, R):
        '''
        Layer type specific propogation of relevance
        '''
        R = R.view(self.output.shape)  # stitching ".view" step, frequantly in classifier
        layer_name = self.module._get_name()
        Ri = globals()[layer_name].relprop(copy_module(self.module),
                                           self.input, R, self.num)
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
    def relprop(module, input_, R, num):
        return R


class Dropout(torch.nn.Dropout):

    @staticmethod
    def relprop(module, input_, R, num):
        return R


class Linear(torch.nn.Linear):

    @staticmethod
    def relprop(module, input_, R, num):
        if num == MODEL_DEPTH - 1: #if last layer
            #R = rules.w2_rule(module, input_, R)
            #R = rules.z_plus_rule(module, input_, R)
            #R = rules.z_rule(module, input_, R)
            R = rules.z_epsilon_rule(module, input_, R)
        elif num == 0: #if first layer
            R = rules.z_box_rule(module, input_, R, lowest=-1,
              highest=1) #image should be maped to [-1, 1] intv
        else:
            #R = rules.z_plus_rule(module, input_, R)
            #R = rules.w2_rule(module, input_, R)
            #R = rules.z_rule(module, input_, R)
            R = rules.z_epsilon_rule(module, input_, R)
        return R


class Conv2d(torch.nn.Conv2d):

    @staticmethod
    def relprop(module, input_, R, num):
        if num == 0:
            R = rules.z_box_rule(module, input_, R, lowest=-1, highest=1)
           #R = rules.z_plus_rule(module, input_, R)
           #R = rules.w2_rule(module, input_, R)
        else: #nextconvolitional layer
            R = rules.z_epsilon_rule(module, input_, R)
            #R = rules.z_rule(module, input_, R)
            #R = rules.w2_rule(module, input_, R)
        return R


class MaxPool2d(torch.nn.MaxPool2d):

    @staticmethod
    def relprop(module, input_, R, num):
        #R = z_plus_rule(module, input_, R)
        #R = w2_rule(module, input_, R)
        R = rules.z_epsilon_rule(module, input_, R)
        #R = rules.z_rule(module, input_, R)
        return R