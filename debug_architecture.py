#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import hiddenlayer as hl
from modelsummary import summary


# In[4]:


#use pip install here to load missing stuff
#! pip install modelsummary
#!pip install torch_summary


# In[8]:


#nb with torch_summary recent version, there is something funny as you have to import torchsummary...
from torchsummary import *


# In[9]:


#well the 9/July/20 with recent Catalina install and so on I get 1.4.0, 
# previously I was running 1.1.0
print(torch.__version__)


# In[14]:


from collections import OrderedDict
import pandas as pd

import torch
from torch import nn
from torch.autograd import Variable

class TorchSummarizeDf(object):
    def __init__(self, model, weights=False, input_shape=True, nb_trainable=False, debug=False):
        """
        Summarizes torch model by showing trainable parameters and weights.

        author: wassname
        url: https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7
        license: MIT

        Modified from:
        - https://github.com/pytorch/pytorch/issues/2001#issuecomment-313735757
        - https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7/

        Usage:
            import torchvision.models as models
            model = models.alexnet()
            # attach temporary hooks using `with`
            with TorchSummarizeDf(model) as tdf:
                x = Variable(torch.rand(2, 3, 224, 224))
                y = model(x)
                df = tdf.make_df()
            print(df)

            # Total parameters 61100840
            #              name class_name        input_shape       output_shape  nb_params
            # 1     features=>0     Conv2d  (-1, 3, 224, 224)   (-1, 64, 55, 55)      23296
            # 2     features=>1       ReLU   (-1, 64, 55, 55)   (-1, 64, 55, 55)          0
            # ...
        """
        # Names are stored in parent and path+name is unique not the name
        self.names = get_names_dict(model)

        # store arguments
        self.model = model
        self.weights = weights
        self.input_shape = input_shape
        self.nb_trainable = nb_trainable
        self.debug = debug

        # create properties
        self.summary = OrderedDict()
        self.hooks = []

    def register_hook(self, module):
        """Register hooks recursively"""
        self.hooks.append(module.register_forward_hook(self.hook))

    def hook(self, module, input, output):
        """This hook is applied when each module is run"""
        class_name = str(module.__class__).split('.')[-1].split("'")[0]
        module_idx = len(self.summary)
        
        name = None
        for key, item in self.names.items():
            if item == module:
                name = key
        if name is None:
            name = '{}_{}'.format(class_name, module_idx)

        m_key = module_idx + 1

        self.summary[m_key] = OrderedDict()
        self.summary[m_key]['name'] = name
        self.summary[m_key]['class_name'] = class_name

        # Handle multiple inputs
        if self.input_shape:
            # for each input remove batch size and replace with one
            self.summary[m_key][
                'input_shape'] = format_input_output_shape(input)

        # Handle multiple outputs
        self.summary[m_key]['output_shape'] = format_input_output_shape(output)

        if self.weights:
            self.summary[m_key]['weights'] = list(
                [tuple(p.size()) for p in module.parameters()])

        if self.nb_trainable:
            self.summary[m_key]['nb_trainable'] = get_params(module, True)
            
        self.summary[m_key]['nb_params'] = get_params(module, True)
        
        if self.debug:
            print(self.summary[m_key])

    def __enter__(self):

        # register hook
        self.model.apply(self.register_hook)

        # make a forward pass
        self.training = self.model.training
        if self.training:
            self.model.eval()

        return self

    def make_df(self):
        """Make dataframe."""
        df = pd.DataFrame.from_dict(self.summary, orient='index')

        df['level'] = df['name'].apply(lambda name: name.count('.'))
        
        total_params = get_params(self.model, False)
        total_trainable_params = get_params(self.model, True)
        print('Total parameters', total_params)
        print('Total trainable parameters', total_trainable_params)
        return df

    def __exit__(self, exc_type, exc_val, exc_tb):

        if exc_type or exc_val or exc_tb:
            # to help with debugging your model lets print the summary even if it fails
            df_summary = pd.DataFrame.from_dict(self.summary, orient='index')
            print(df_summary)

        if self.training:
            self.model.train()

        # remove these hooks
        for h in self.hooks:
            h.remove()


def get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=''):
        for key, module in module.named_children():
            name = parent_name + '.' + key if parent_name else key
            names[name] = module
            if isinstance(module, torch.nn.Module):
                _get_names(module, parent_name=name)
    _get_names(model)
    return names

def get_params(module, nb_trainable=False):
    if nb_trainable:
        params = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters() if p.requires_grad])
    else:
        params = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters()])
    if isinstance(params, torch.Tensor):
        params = params.item()
    return params

def format_input_output_shape(tensors):
    "Recursively get N nested levels of inputs."""
    def _format_input_output_shape(tensors):
        if isinstance(tensors, (list, tuple)):
            if len(tensors)==1:
                return _format_input_output_shape(tensors[0])
            else:
                return [_format_input_output_shape(tensor) for tensor in tensors]
        else:
            return [(-1, ) + tuple(tensors.size())[1:]]
    return _format_input_output_shape(tensors)


# In[10]:


class PzConv2d(nn.Module):
    """ Convolution 2D Layer followed by PReLU activation
    """
    def __init__(self, n_in_channels, n_out_channels, **kwargs):
        super(PzConv2d, self).__init__()
        self.conv = nn.Conv2d(n_in_channels, n_out_channels, bias=True,
                            **kwargs)
        ## JEC 11/9/19 use default init :
        ##   kaiming_uniform_ for weights
        ##   bias uniform 
        #xavier init for the weights
        ## nn.init.xavier_normal_(self.conv.weight)
        nn.init.xavier_uniform_(self.conv.weight)
        ## constant init for the biais with cte=0.1
        nn.init.constant_(self.conv.bias,0.1)
####        self.bn = nn.BatchNorm2d(n_out_channels, eps=0.001)  #### TEST JEC 4/11/19 for robust training
        self.activ = nn.PReLU(num_parameters=n_out_channels, init=0.25)
        ## self.activ = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
####        x = self.bn(x) #### TEST JEC 4/11/19 for robust training
        return self.activ(x)


class PzPool2d(nn.Module):
    """ Average Pooling Layer
    """
    def __init__(self, kernel_size, stride, padding=0):
        super(PzPool2d, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 ceil_mode=True,
                                 count_include_pad=False)

    def forward(self, x):
        return self.pool(x)


class PzFullyConnected(nn.Module):
    """ Dense or Fully Connected Layer followed by ReLU
    """
    def __init__(self, n_inputs, n_outputs, withrelu=True, **kwargs):
        super(PzFullyConnected, self).__init__()
        self.withrelu = withrelu
        self.linear = nn.Linear(n_inputs, n_outputs, bias=True)
        ## JEC 11/9/19 use default init :
        ##   kaiming_uniform_ for weights
        ##   bias uniform 

        # xavier init for the weights
        nn.init.xavier_uniform_(self.linear.weight)
##        nn.init.xavier_normal_(self.linear.weight)
        # constant init for the biais with cte=0.1
        nn.init.constant_(self.linear.bias, 0.1)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        if self.withrelu:
            x = self.activ(x)
        return x


class PzInception(nn.Module):
    """ Inspection module

        The input (x) is dispatched between

        o a cascade of conv layers s1_0 1x1 , s2_0 3x3
        o a cascade of conv layer s1_2 1x1, followed by pooling layer pool0 2x2
        o a cascade of conv layer s2_2 1x1
        o optionally a cascade of conv layers s1_1 1x1, s2_1 5x5

        then the 3 (or 4) intermediate outputs are concatenated
    """
    def __init__(self, n_in_channels, n_out_channels_1, n_out_channels_2,
                 without_kernel_5=False, debug=False):
        super(PzInception, self).__init__()
        self.debug = debug
        self.s1_0 = PzConv2d(n_in_channels, n_out_channels_1,
                             kernel_size=1, padding=0)
        self.s2_0 = PzConv2d(n_out_channels_1, n_out_channels_2,
                             kernel_size=3, padding=1)

        self.s1_2 = PzConv2d(n_in_channels, n_out_channels_1, kernel_size=1)
        self.pad0 = nn.ZeroPad2d([0, 1, 0, 1])
        self.pool0 = PzPool2d(kernel_size=2, stride=1, padding=0)

        self.without_kernel_5 = without_kernel_5
        if not (without_kernel_5):
            self.s1_1 = PzConv2d(n_in_channels, n_out_channels_1,
                                 kernel_size=1, padding=0)
            self.s2_1 = PzConv2d(n_out_channels_1, n_out_channels_2,
                                 kernel_size=5, padding=2)

        self.s2_2 = PzConv2d(n_in_channels, n_out_channels_2, kernel_size=1,
                             padding=0)

    def forward(self, x):
        # x:image tenseur N_batch, Channels, Height, Width
        x_s1_0 = self.s1_0(x)
        x_s2_0 = self.s2_0(x_s1_0)

        x_s1_2 = self.s1_2(x)

        x_pool0 = self.pool0(self.pad0(x_s1_2))

        if not (self.without_kernel_5):
            x_s1_1 = self.s1_1(x)
            x_s2_1 = self.s2_1(x_s1_1)

        x_s2_2 = self.s2_2(x)

        if self.debug: print("Inception x_s1_0  :", x_s1_0.size())
        if self.debug: print("Inception x_s2_0  :", x_s2_0.size())
        if self.debug: print("Inception x_s1_2  :", x_s1_2.size())
        if self.debug: print("Inception x_pool0 :", x_pool0.size())

        if not (self.without_kernel_5) and self.debug:
            print("Inception x_s1_1  :", x_s1_1.size())
            print("Inception x_s2_1  :", x_s2_1.size())

        if self.debug: print("Inception x_s2_2  :", x_s2_2.size())

        # to be check: dim=1=> NCHW (en TensorFlow axis=3 NHWC)
        if not (self.without_kernel_5):
            output = torch.cat((x_s2_2, x_s2_1, x_s2_0, x_pool0), dim=1)
        else:
            output = torch.cat((x_s2_2, x_s2_0, x_pool0), dim=1)

        if self.debug: print("Inception output :", output.shape)
        return output


class NetWithInception(nn.Module):
    """ The Networks
        inputs: the image (x), the reddening vector


        The image 64x64x5 is fed forwardly throw
        o a conv layer 5x5
        o a pooling layer 2x2
        o 5 inspection modules with the last one including a 5x5 part

        Then, we concatenate the result with the reddening vector to perform
        o 3 fully connected layers

        The output dimension is given by n_bins
        There is no activation softmax here to allow the use of Cross Entropy loss

    """
    def __init__(self, n_input_channels, debug=False):
        super(NetWithInception, self).__init__()
        
        # the number of bins to represent the output photo-z
        self.n_bins = 180

        self.debug = debug
        self.conv0 = PzConv2d(n_in_channels=n_input_channels,
                              n_out_channels=64,
                              kernel_size=5, padding=2)
        self.pool0 = PzPool2d(kernel_size=2, stride=2, padding=0)
        # for the Softmax the input tensor shape is [1,n] so apply on axis=1
        # t1 = torch.rand([1,10])
        # t2 = nn.Softmax(dim=1)(t1)
        # torch.sum(t2) = 1
        self.i0 = PzInception(n_in_channels=64,
                              n_out_channels_1=48,
                              n_out_channels_2=64)

        self.i1 = PzInception(n_in_channels=240,
                              n_out_channels_1=64,
                              n_out_channels_2=92)

        self.i2 = PzInception(n_in_channels=340,
                              n_out_channels_1=92,
                              n_out_channels_2=128)

        self.i3 = PzInception(n_in_channels=476,
                              n_out_channels_1=92,
                              n_out_channels_2=128)

        self.i4 = PzInception(n_in_channels=476,
                              n_out_channels_1=92,
                              n_out_channels_2=128,
                              without_kernel_5=True)

        self.fc0 = PzFullyConnected(n_inputs=22273, n_outputs=1096)
        self.fc1 = PzFullyConnected(n_inputs=1096, n_outputs=1096)
        self.fc2 = PzFullyConnected(n_inputs=1096, n_outputs=self.n_bins)


        # NOT USED self.activ = nn.Softmax(dim=1)

    def num_flat_features(self, x):
        """

        Parameters
        ----------
        x: the input

        Returns
        -------
        the totale number of features = number of elements of the tensor except the batch dimension

        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x, reddening):
        # x:image tenseur N_batch, Channels, Height, Width
        #    size N, Channles=5 filtres, H,W = 64 pixels
        # save original image
        x_in = x

        if self.debug: print("input shape: ", x.size())
        x = self.conv0(x)
        if self.debug: print("conv0 shape: ", x.size())
        x = self.pool0(x)
        if self.debug: print("conv0p shape: ", x.size())
        if self.debug: print('>>>>>>> i0:START <<<<<<<')
        x = self.i0(x)
        if self.debug: print("i0 shape: ", x.size())

        if self.debug: print('>>>>>>> i1:START <<<<<<<')
        x = self.i1(x)

        x = self.pool0(x)
        if self.debug: print("i1p shape: ", x.size())

        if self.debug: print('>>>>>>> i2:START <<<<<<<')
        x = self.i2(x)
        if self.debug: print("i2 shape: ", x.size())

        if self.debug: print('>>>>>>> i3:START <<<<<<<')
        x = self.i3(x)
        x = self.pool0(x)
        if self.debug: print("i3p shape: ", x.size())

        if self.debug: print('>>>>>>> i4:START <<<<<<<')
        x = self.i4(x)
        if self.debug: print("i4 shape: ", x.size())

        if self.debug: print('>>>>>>> FC part :START <<<<<<<')
        flat = x.view(-1, self.num_flat_features(x))
        if self.debug: print("flat shape: ", flat.size())
        concat = torch.cat((flat, reddening), dim=1)
        if self.debug: print('concat shape: ', concat.size())

        fcn_in_features = concat.size(-1)
        if self.debug: print('fcn_in_features: ', fcn_in_features)

        x = self.fc0(concat)
        if self.debug: print('fc0 shape: ', x.size())
        x = self.fc1(x)
        if self.debug: print('fc1 shape: ', x.size())
        x = self.fc2(x)
        if self.debug: print('fc2 shape: ', x.size())

        output = x
        if self.debug: print('output shape: ', output.size())

        #params = {"output": output, "x": x_in, "reddening": reddening}
        # return params

        return output


# In[11]:


img_channels = 5
img_H = 64
img_W = 64
n_batchs = 1
model = NetWithInception(img_channels,debug=False)
#Notice that if dtype=torch.double ca plente
imgs = torch.zeros([n_batchs, img_channels,img_H ,img_W],dtype=torch.float)
reds = torch.zeros([n_batchs,1],dtype=torch.float)
print("imgs type ",imgs)
print("reds type ",reds)
model(imgs,reds)


# In[12]:


print(model)


# In[15]:


# attach temporary hooks using `with`
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
with TorchSummarizeDf(model) as tdf:
    imgs = torch.zeros([n_batchs, img_channels,img_H ,img_W])
    reds = torch.zeros([n_batchs,1])
    y = model(imgs,reds)
    df = tdf.make_df()
    print(df)


# In[ ]:




