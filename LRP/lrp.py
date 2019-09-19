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
        if R is None: #if input R (relevance) is None select max logit
            R = (self.output == self.output.max()).float()
        self.output.backward(R, retain_graph=True)
        relevance = self.local_input.grad
        assert relevance is not None, 'obtained relevance is None'
        return relevance

    __call__ = forward


