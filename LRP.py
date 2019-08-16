import torch
import copy
import pickle

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
    return pickle.loads(pickle.dumps(module))

class LRP():
    '''
    torch implementation of LRP
    http://www.heatmapping.org/tutorial/
    '''
    def __init__(self, model):
        self.model = copy.deepcopy(model)
        self.model = self.model.eval()
        #put hook to each basic layer
        self.hooks = [Hook(num, module) for num, module
                      in enumerate(flatten_model(self.model))]
        self.output = None

    def forward(self, input_):
        self.output = self.model(input_)
        return self.output

    def relprop(self, R):
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
        self.num = num# serial number of basic layer
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
        LAyer type specific propogation of relevance
        '''
        R = R.view(self.output.shape) # stitching ".view" step, frequantly in classifier
        layer_name = self.module._get_name()
        Ri = globals()[layer_name].relprop(copy_module(self.module), self.input, R, self.num)
       # if layer_name == 'Conv2d' and self.num != 0:
       #     Ri = NextConvolution2d.relprop(copy_module(self.module), self.input, R)
       # if layer_name == 'Conv2d' and self.num == 0:
       #     Ri = FirstConvolution2d.relprop(copy_module(self.module), self.input, R)
       # elif layer_name == 'Dropout':
       #     Ri = Dropout.relprop(copy_module(self.module), self.input, R)
       # elif layer_name == 'Linear' and self.num != 0:
       #     Ri = NextLinear.relprop(copy_module(self.module), self.input, R)
       # elif layer_name == 'MaxPool2d':
       #     Ri = MaxPool2d.relprop(copy_module(self.module), self.input, R)
       # elif layer_name == 'ReLU':
       #     Ri = ReLU.relprop(copy_module(self.module), self.input, R)
        if self.input.grad is not None:
            self.input.grad.zero_()
        return Ri.detach()

    def __del__(self):
        self.close()
        del_super = getattr(super(Hook, self), "__del__", None)
        if callable(del_super):
            del_super()

#Initial idea was to overload backward pass, but then I stick with hooks
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
        if num !=0 :
            pself = module
            pself.bias.data.zero_()
            pself.weight.data.clamp_(0, float('inf'))
            Z = pself(input_)+1e-9
            S = R/Z
            Z.backward(S)
            C = input_.grad
            R = input_*C
        elif num == 0:
            raise NotImplementedError
        #DOTO implement first linear layer
        return R

class Conv2d(torch.nn.Conv2d):

    @staticmethod
    def relprop(module, input_, R, num):
        if num != 0: #nextconvolitional layer
            pself = module
            pself.bias.data.zero_()
            pself.weight.data.clamp_(0, float('inf'))
            Z = pself(input_)+1e-9
            S = R/Z
            Z.backward(S)
            C = input_.grad
            R = input_*C
        elif num == 0:
            lowest = -0.5 #input_.min()
            highest = 3 #input_.max()
            iself = copy.deepcopy(module)
            iself.bias.data.zero_()
            nself = copy.deepcopy(module)
            nself.bias.data.zero_()
            nself.weight.data.clamp_(-float('inf'), 0)
            pself = copy.deepcopy(module)
            pself.bias.data.zero_()
            pself.weight.data.clamp_(0, float('inf'))
            L = torch.zeros_like(input_)+lowest
            H = torch.zeros_like(input_)+highest
            L.requires_grad_(True)
            L.retain_grad()
            H.requires_grad_(True)
            H.retain_grad()
            Z = iself(input_)-pself(L)-nself(H)+1e-9
            S = R/Z
            Z.backward(S)
            R = input_*input_.grad-L*L.grad-H*H.grad
        return R

class MaxPool2d(torch.nn.MaxPool2d):

    @staticmethod
    def relprop(module, input_, R, num):
        Z = module(input_) + 1e-9
        S = R/Z
        Z.backward(S)
        R = input_.grad*input_
        return R
