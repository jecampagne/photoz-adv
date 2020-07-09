import torch
import torch.nn as nn


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

    
## ###############

class NetCNN(nn.Module):
    def __init__(self,n_input_channels,debug=False):
        super(NetCNN, self).__init__()

        # the number of bins to represent the output photo-z
        self.n_bins = 180


        self.debug = debug
        self.conv0 = PzConv2d(n_in_channels=n_input_channels,
                              n_out_channels=64,
                              kernel_size=5,padding=2)
        self.pool0 = PzPool2d(kernel_size=2,stride=2,padding=0)

        self.conv1 = PzConv2d(n_in_channels=64,
                              n_out_channels=92,
                              kernel_size=3,padding=2)
        self.pool1 = PzPool2d(kernel_size=2,stride=2,padding=0)

        self.conv2 = PzConv2d(n_in_channels=92,
                              n_out_channels=128,
                              kernel_size=3,padding=2)
        self.pool2 = PzPool2d(kernel_size=2,stride=2,padding=0)
        
        
        self.fc0 = PzFullyConnected(n_inputs=12800,n_outputs=1024)
        self.fc1 = PzFullyConnected(n_inputs=1024,n_outputs=self.n_bins)
                
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
  
    def forward(self, x, reddening):
        # x:image tenseur N_batch, Channels, Height, Width
        #    size N, Channles=5 filtres, H,W = 64 pixels
        # reddening: is not used
      
        #save original image
##        x_in = x
      
        if self.debug: print("input shape: ",x.size())
        
        # stage 0 conv 64 x 5x5
        x = self.conv0(x)
        if self.debug: print("conv0 shape: ",x.size())
        x = self.pool0(x)
        if self.debug: print("conv0p shape: ",x.size()) 

        # stage 1 conv 92 x 3x3
        x = self.conv1(x)
        if self.debug: print("conv1 shape: ",x.size())
        x = self.pool1(x)
        if self.debug: print("conv1p shape: ",x.size()) 
 
        # stage 2 conv 128 x 3x3
        x = self.conv2(x)
        if self.debug: print("conv2 shape: ",x.size())
        x = self.pool2(x)
        if self.debug: print("conv2p shape: ",x.size()) 
            
        
        if self.debug: print('>>>>>>> FC part :START <<<<<<<')        
        x = self.fc0(x.view(-1,self.num_flat_features(x)))
        if self.debug: print('fc0 shape: ',x.size())
        x = self.fc1(x)
        if self.debug: print('fc1 shape: ', x.size())

        output = x
      
        if self.debug: print('output shape: ',output.size())
      
##        params = {"output": output, "x": x_in, "reddening": reddening}
        
        #return params           
        return output

## ##################

class NetCNNRed(nn.Module):
    def __init__(self,n_input_channels,debug=False):
        super(NetCNNRed, self).__init__()
        self.debug = debug
        # the number of bins to represent the output photo-z
        self.n_bins = 180

        self.conv0 = PzConv2d(n_in_channels=n_input_channels,
                              n_out_channels=64,
                              kernel_size=5,padding=2)
        self.pool0 = PzPool2d(kernel_size=2,stride=2,padding=0)

        self.conv1 = PzConv2d(n_in_channels=64,
                              n_out_channels=92,
                              kernel_size=3,padding=2)
        self.pool1 = PzPool2d(kernel_size=2,stride=2,padding=0)

        self.conv2 = PzConv2d(n_in_channels=92,
                              n_out_channels=128,
                              kernel_size=3,padding=2)
        self.pool2 = PzPool2d(kernel_size=2,stride=2,padding=0)


        self.fc0 = PzFullyConnected(n_inputs=12801,n_outputs=1024)
        self.fc1 = PzFullyConnected(n_inputs=1024,n_outputs=180)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x, reddening):
        # x:image tenseur N_batch, Channels, Height, Width
        #    size N, Channles=5 filtres, H,W = 64 pixels
        # reddening: used

        #save original image
##        x_in = x

        if self.debug: print("input shape: ",x.size())

        # stage 0 conv 64 x 5x5
        x = self.conv0(x)
        if self.debug: print("conv0 shape: ",x.size())
        x = self.pool0(x)
        if self.debug: print("conv0p shape: ",x.size()) 

        # stage 1 conv 92 x 3x3
        x = self.conv1(x)
        if self.debug: print("conv1 shape: ",x.size())
        x = self.pool1(x)
        if self.debug: print("conv1p shape: ",x.size()) 

        # stage 2 conv 128 x 3x3
        x = self.conv2(x)
        if self.debug: print("conv2 shape: ",x.size())
        x = self.pool2(x)
        if self.debug: print("conv2p shape: ",x.size()) 


        if self.debug: print('>>>>>>> FC part :START <<<<<<<')
        flat   = x.view(-1,self.num_flat_features(x))
        concat = torch.cat((flat,reddening),dim=1)
        if self.debug: print('concat shape: ', concat.size())

        x = self.fc0(concat)
        if self.debug: print('fc0 shape: ',x.size())
        x = self.fc1(x)
        if self.debug: print('fc1 shape: ', x.size())

        output = x

        if self.debug: print('output shape: ',output.size())

##        params = {"output": output, "x": x_in, "reddening": reddening}

        #return params           
        return output

