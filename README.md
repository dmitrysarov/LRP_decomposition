This repo consist of LRP (Layer wise propagation) implementation in pytorch.
Here you can find nice but in keras implementation
https://github.com/sebastian-lapuschkin/lrp_toolbox

This implementation use custom autograd functions to make possiable pass though
relevance in backward pass instead gradients. This gives ability to propagate
relevance from skip connections (opposite to sequantial approach from another
branch where i am using hooks)

LRP_notebook.ipynb - demonstration on MNIST data

find_root.ipynb - root point calculation notebook

mnist_model.ph - pretrained weights from simple conv model

mnist_model.py - script to obtain simple conv model
