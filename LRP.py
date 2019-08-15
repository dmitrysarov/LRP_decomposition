import torch
import copy

def flatten_model(module):
    modules_list = []
    for m_1 in module.children():
        if len(list(m_1.children())) == 0:
            modules_list.append(m_1)
        else:
            modules_list = modules_list + flatten_model(m_1)
    return modules_list

class LRP():
    '''
    torch implementation of LRP
    http://www.heatmapping.org/tutorial/
    '''
    def __init__(self, model):
        self.model = model.eval()
        # obtain base moduls (not combination like Sequantial) and add hook to it
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

class Hook():
    '''
    Hook will be add to each component of NN
    DOTO get ordered according to forwardpass layer list https://discuss.pytorch.org/t/list-layers-nodes-executed-in-order-of-forward-pass/46610
    '''
    def __init__(self, num, module):
        self.num = num# serial number of "layer" (component)
        self.module = copy.deepcopy(module)
        self.hookF = module.register_forward_hook(self.hook_fnF)
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
        self.hook.remove()

    def relprop(self, R):
        '''
        Component type specific propogation of relevance
        '''
        R = R.view(self.output.shape) # stitching ".view" step, frequantly in classifier
        layer_name = self.module._get_name()
        if layer_name == 'Conv2d' and self.num != 0:
            Ri = NextConvolution2d.relprop(self, R)
        if layer_name == 'Conv2d' and self.num == 0:
            Ri = FirstConvolution2d.relprop(self, R)
        elif layer_name == 'Dropout':
            Ri = Dropout.relprop(self, R)
        elif layer_name == 'Linear' and self.num != 0:
            Ri = NextLinear.relprop(self, R)
        elif layer_name == 'MaxPool2d':
            Ri = MaxPool2d.relprop(self, R)
        elif layer_name == 'ReLU':
            Ri = ReLU.relprop(self, R)
        return Ri.detach()

#Initial idea was to overload backward pass, but then I stick with hooks
class ReLU(torch.nn.ReLU):

    @staticmethod
    def relprop(self,R):
        return R

class Dropout(torch.nn.Dropout):

    @staticmethod
    def relprop(self,R):
        return R

class NextLinear(torch.nn.Linear):

    @staticmethod
    def relprop(self,R):
        pself = self.module
        pself.bias.data.zero_()
        pself.weight.data.clamp_(0, float('inf'))
        Z = pself(self.input)+1e-9
        S = R/Z
        Z.backward(S)
        C = self.input.grad
        R = self.input*C
        return R

class NextConvolution2d(torch.nn.Conv2d):

    @staticmethod
    def relprop(self,R):
        pself = self.module
        pself.bias.data.zero_()
        pself.weight.data.clamp_(0, float('inf'))
        Z = pself(self.input)+1e-9
        S = R/Z
        Z.backward(S)
        C = self.input.grad
        R = self.input*C
        return R

class FirstConvolution2d(torch.nn.Conv2d):

    @staticmethod
    def relprop(self,R):
        lowest = -0.5 #self.input.min()
        highest = 3 #self.input.max()
        iself = copy.deepcopy(self.module)
        iself.bias.data.zero_()
        nself = copy.deepcopy(self.module)
        nself.bias.data.zero_()
        nself.weight.data.clamp_(-float('inf'), 0)
        pself = copy.deepcopy(self.module)
        pself.bias.data.zero_()
        pself.weight.data.clamp_(0, float('inf'))
        L = torch.zeros_like(self.input)+lowest
        H = torch.zeros_like(self.input)+highest
        L.requires_grad_(True)
        L.retain_grad()
        H.requires_grad_(True)
        H.retain_grad()
        Z = iself(self.input)-pself(L)-nself(H)+1e-9
        S = R/Z
        Z.backward(S)
        R = self.input*self.input.grad-L*L.grad-H*H.grad
        return R

class MaxPool2d(torch.nn.MaxPool2d):

    @staticmethod
    def relprop(self,R):
        Z = self.module(self.input) + 1e-9
        S = R/Z
        Z.backward(S)
        R = self.input.grad*self.input
        return R
