import torch
from  torch.nn import modules
import torch.nn.functional as F

def __getattr__(name):
    '''
    to get layer modules classes as its attribute
    '''
    if name in globals():
        return globals()[name]
    else:
        return AttributeError

#Definition of custom autograd function
class LRP_zrule_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func, input, func_args, rule_func):
        '''
        forawd pass perform usual func forward pass with input(tensor) and func_args(dict) as arguments
        rule(string) is the name of chosen rule
        '''
        ctx.func = func
        ctx.input =input.clone().detach()
        ctx.func_args = func_args
        ctx.rule_func = rule_func
        return func(input, **func_args)

    @staticmethod
    def backward(ctx, R):
        '''
        substitute backward pass with z rule propagation
        backward pass must return same namber of ouputs as number of inputs in forward pass
        '''
       # ctx.input.requires_grad_(True)
       # with torch.enable_grad():
       #     Z = ctx.func(ctx.input, *ctx.args)
       #     S = R /(Z + (Z==0).float()*np.finfo(np.float32).eps)
       #     Z.backward(S)
       #     C = ctx.input.grad
       #     R = ctx.input * C
        R = ctx.rule_func(ctx.func, ctx.input, R, ctx.func_args)

        return None, R, None, None

class LRP_relu_func(torch.autograd.Function):
    '''
    perform simple pass of relevance during backward
    '''
    @staticmethod
    def forward(ctx, func, input, args):
        ctx.args = args
        return func(input, **args)

    @staticmethod
    def backward(ctx, R):
        return None, R, None

# Dumb classes just for convinient assignment (overload) of model layers
# methods
class Conv2d(object):
    def conv2d_forward(self, input, weight):
        ##Test
        #print(self.__class__.__name__)
        if self.padding_mode == 'circular':
            NotImplementedError
       #     expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
       #                         (self.padding[0] + 1) // 2, self.padding[0] // 2)
       #     return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
       #                     weight, self.bias, self.stride,
       #                     _pair(0), self.dilation, self.groups) #TODO _pair(0)?
        return LRP_zrule_func.apply(F.conv2d, input, {'weight': weight, 'bias': self.bias,
            'stride': self.stride, 'padding': self.padding, 'dilation': self.dilation,
            'groups': self.groups}, self.rule_func)
       # if self.padding_mode == 'circular':
       #     expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
       #                         (self.padding[0] + 1) // 2, self.padding[0] // 2)
       #     return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
       #                     weight, self.bias, self.stride,
       #                     _pair(0), self.dilation, self.groups)
       # return F.conv2d(input, weight, self.bias, self.stride,
       #                 self.padding, self.dilation, self.groups)

class BatchNorm2d(object):
    def forward(self, input):
        ##Test
        #print(self.__class__.__name__)
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

class ReLU(object):
    def forward(self, input):
        ##Test
        #print(self.__class__.__name__)
        return LRP_relu_func.apply(F.relu, input, {'inplace': self.inplace})

class Linear(object):
    def forward(self, input):
        return LRP_zrule_func.apply(F.linear, input, {'weight': self.weight, 'bias': self.bias}, self.rule_func)
