from torchvision.models import resnet18
import torch
import imp
utils = imp.load_source('util', '../utils.py')
flatten_model = utils.flatten_model

model = resnet18()
out = model(torch.ones(1,3,256,256))
