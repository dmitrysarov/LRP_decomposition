This repo consist of LRP (Layer wise propagation) implementation in pytorch.
Here you can find nice but in keras implementation
https://github.com/sebastian-lapuschkin/lrp_toolbox

This implementation use custom autograd functions to make possiable pass though
relevance in backward pass instead gradients. This gives ability to propagate
relevance from skip connections (opposite to sequantial approach from another
branch where i am using hooks)

Personally I was curious to implement resnet LRP.
I found one pytorch LRP implementation here
https://github.com/moboehle/Pytorch-LRP . But implementation principally 
is the same as mine through hooks - sequantial propagation from NN.
The quiastion is - is it valid for skip connection. I guess not.
Current implemention could be used for resnet with little hack - needness of
substitube identity summation operation with custom autograd function.

I overload layers forward pass, where substitute default autograd
function with custom one, excepting as argument - default function, input,
default function arguments, and LRP_rule function

LRP_notebook.ipynb - demonstration on MNIST data

find_root.ipynb - root point calculation notebook

mnist_model.ph - pretrained weights from simple conv model

mnist_model.py - script to obtain simple conv model
