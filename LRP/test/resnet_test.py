import torch
from torchvision.models import resnet18
import imp
LRP = imp.load_source('LRP', '../__init__.py')
from LRP import utils

model = resnet18()
model = utils.redefine_nn(model)
input = torch.rand(1,3,256,256) 
input.requires_grad_(True)
input.register_hook(lambda x: print(x.shape))
out = model(input)
out.backward(out)
