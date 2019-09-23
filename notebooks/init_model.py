import torch

class MMODEL(torch.nn.Module):
    def __init__(self):
        super(MMODEL, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1,1,(1,1)),
            torch.nn.MaxPool2d((1,1))
            )
    def forward(self, x):
        return self.model(x)

model = MMODEL()
