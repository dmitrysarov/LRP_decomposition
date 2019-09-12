This repo consist of LRP (Layer wise propagation) implementation in pytorch.
Here you can find nice but in my opinion overengineering keras implementation
https://github.com/sebastian-lapuschkin/lrp_toolbox

Week implemenation because model should be defined in proper way (namely model.children should be sequance of nn.Sequence) and I don't know 
how to implement add operation (e.g. in skip connections) in this paradigm. 
More general approach should be implemented

LRP_notebook.ipynb - demonstration on MNIST data

find_root.ipynb - root point calculation notebook

mnist_model.ph - pretrained weights from simple conv model

mnist_model.py - script to obtain simple conv model
