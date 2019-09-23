import torch

class MCONV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, layer):
        import ipdb; ipdb.set_trace() # BREAKPOINT
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
def wrap(input, layer)
layer = torch.nn.Conv2d(1,1,(1,1))
input_ = torch.tensor([1], dtype=torch.float32).view(1,1,1,1)
layer= MCONV(layer).apply
output = layer(input_)
output.backward()
