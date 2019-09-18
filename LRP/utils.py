import copy
import pickle
import torch
from . import layers

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
    module._backward_hooks.popitem()  # remove hooks from module copy
    return module

def redefine_nn(model):
    '''
    go over model layers and overload choosen methods (e.g. forward()).
    New methods come from classes of layers module
    '''
    list_of_layers = dir(layers) #list of redefined layers in layers module
    for module in flatten_model(model):
        if module.__class__.__name__ in list_of_layers:
            local_class = module.__class__ #current layer class
            layer_module_class = layers.__getattr__(local_class.__name__) #redefine layer class
            list_of_methods = [attr for attr in dir(layer_module_class) if attr[:2] != '__'] #methods which  was redefined
            for l in list_of_methods:
                setattr(local_class, l, getattr(layer_module_class, l)) #set redefined methods to layer class
    return model
##Test
#from torchvision.models import resnet18
#model = resnet18()
#model = redefine_nn(model)
#model(torch.rand(1,3,256,256))
