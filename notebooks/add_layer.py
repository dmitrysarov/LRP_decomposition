import torch
import copy
import imp
utils = imp.load_source('util', '../utils.py')
flatten_model = utils.flatten_model

class LRP():
    '''
    torch implementation of lrp
    http://www.heatmapping.org/tutorial/
    :param rule:str: name of used rule
    '''

    def __init__(self, model):
        self.model = copy.deepcopy(model)
        self.model = self.model.eval()
        # put hook to each basic layer
        self.hooks = [Hook(module) for num, module
                      in enumerate(utils.flatten_model(self.model))]
        self.output = None
    #    self.rule = rules.__getattr__(rule) #rule - str name of rule

    def forward(self, input_):
        self.output = self.model(input_)
        return self.output

    def relprop(self, r):
       # global model_depth
       # model_depth = len(self.hooks)
        assert self.output is not none, 'forward pass not performed. do .forward() first'
        self.output.backward()
       # for h in self.hooks[::-1]:
       #     r = h.relprop(r, self.rule)
        return r

    __call__ = forward

class Hook(object):
    '''
    Hook will be add to each component of NN
    DOTO get ordered according to forwardpass layer list https://discuss.pytorch.org/t/list-layers-nodes-executed-in-order-of-forward-pass/46610
    '''

    def __init__(self, module):
        #self.num = num  # serial number of basic layer
        self.module = module
        self.hookF = self.module.register_forward_hook(self.hook_fnF)
        self.hookB = self.module.register_backward_hook(self.hook_fnB)

    def hook_fnF(self, module, input_, output):
        import pudb; pudb.set_trace() # BREAKPOINT
        assert len(input_) == 1, 'batch size should be eaual to 1'
        self.input = copy.deepcopy(input_[0].clone().detach())
        self.input.requires_grad_(True)
        self.output = output

    def hook_fnB(self, module, input, output):
        import pudb; pudb.set_trace() # BREAKPOINT
        self.input = input
        self.output = output

    def close(self):
        self.hookF.remove()

   # def relprop(self, R, rule):
   #     '''
   #     Layer type specific propogation of relevance
   #     '''
   #     R = R.view(self.output.shape)  # stitching ".view" step, frequantly in classifier
   #     layer_name = self.module._get_name()
   #     Ri = globals()[layer_name].relprop(utils.copy_module(self.module),
   #                                        self.input, R, self.num, rule)
   #     Ri = Ri.clone().detach()
   #     if self.input.grad is not None:
   #         self.input.grad.zero_()
   #     return Ri
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layers = torch.nn.Sequential(
                torch.nn.Linear(2,2),
                torch.nn.ReLU(),
                torch.nn.Linear(2,1)
                )
    def forward(self, x):
        return self.layers(x)

model = LRP(SimpleNet())
input = torch.tensor([1,0], dtype=torch.float32).view(1,1,-1)
output = model(input)
print(output)
import pudb; pudb.set_trace() # BREAKPOINT
output.backward()
