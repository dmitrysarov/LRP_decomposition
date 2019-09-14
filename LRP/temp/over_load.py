from init_model import model
import torch
import pprint


#print(model)
#pprint.pprint(model.model[0].__dir__())

conv2d_class = torch.nn.Conv2d

conv2d_class.foo = lambda x: print('hello world')


#pprint.pprint(model.model[0].__dir__())
model.model[0].foo()
