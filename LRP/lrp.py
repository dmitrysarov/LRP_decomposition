import copy
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
        self.model = utils.redefine_nn(self.model, rule=rule) #redefine each layer(module) of model, to set custom autograd func
        self.output = None

    def forward(self, input_):
        self.local_input = input_.clone().detach()
        self.local_input.requires_grad_(True)
        self.output = self.model(self.local_input)
        return self.output.clone().detach()

    def relprop(self, R=None):
        assert self.output is not None, 'First performe forward pass'
        if R is None: #if input R (relevance) is None select max logit
            R = (self.output == self.output.max()).float()
        self.output.backward(R, retain_graph=True)
        C = self.local_input.grad.clone().detach()
        assert C is not None, 'obtained relevance is None'
        self.local_input.grad = None
        return C*self.local_input.clone().detach()

    __call__ = forward


