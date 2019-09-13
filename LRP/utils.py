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
    module = copy.deepcopy(pickle.loads(pickle.dumps(module)))
    module._forward_hooks.popitem()  # remove hooks from module copy
    return module

