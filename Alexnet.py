# Model structure for CNN
from torch.nn import Module # model class
from torch.nn import Conv2d # Convolution
from torch.nn import MaxPool2d # Pooling
from torch.nn import Linear # Dense, fully connected
from torch.nn import LocalResponseNorm
from torch.nn import ReLU # Activation 1
from torch.nn import Tanh # Activation 2
from torch.nn import Dropout # Dropout
import torch

### Model
class Alexnet(Module):
    def __init__(self,
                 height,
                 width,
                 channels
                 ):
        super(Alexnet, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        # Input Shape
        # torch.Size([batch_size, channels, height, width])
        cur_height = self.height
        cur_width = self.width

        ### Convolution 1
        cur_inchannels = self.channels
        cur_outchannels = 96
        cur_kernel = 11
        cur_stride = 4
        cur_dilation = 1
        cur_padding = 0        
        self.conv1 = Conv2d(in_channels=cur_inchannels,
                            out_channels=cur_outchannels,
                            kernel_size=cur_kernel,
                            stride=cur_stride,
                            dilation=cur_dilation,
                            padding=cur_padding)
        self.conv1_act = ReLU()
        cur_height = ((cur_height + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1
        cur_width = ((cur_width + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1
        
        ### Pool 1
        cur_kernel = 3
        cur_stride = 2
        cur_dilation = 1
        cur_padding = 0 
        self.conv1_pool = MaxPool2d(kernel_size=cur_kernel,
                                    stride=cur_stride,
                                    dilation=cur_dilation,
                                    padding=cur_padding)
        cur_height = ((cur_height + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1
        cur_width = ((cur_width + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1
        
        ### Convolution 2
        cur_inchannels = cur_outchannels
        cur_outchannels = 256
        cur_kernel = 5
        cur_stride = 1
        cur_dilation = 1
        cur_padding = 0       
        self.conv1 = Conv2d(in_channels=cur_inchannels,
                            out_channels=cur_outchannels,
                            kernel_size=cur_kernel,
                            stride=cur_stride,
                            dilation=cur_dilation,
                            padding=cur_padding)
        self.conv1_act = ReLU()
        cur_height = ((cur_height + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1
        cur_width = ((cur_width + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1
        
        ### Pool 2
        cur_kernel = 3
        cur_stride = 2
        cur_dilation = 1
        cur_padding = 0
        self.conv1_pool = MaxPool2d(kernel_size=cur_kernel,
                                    stride=cur_stride,
                                    dilation=cur_dilation,
                                    padding=cur_padding)
        cur_height = ((cur_height + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1
        cur_width = ((cur_width + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1



        ### Convolution 3-1
        cur_inchannels = cur_outchannels
        cur_outchannels = 384
        cur_kernel = 3
        cur_stride = 1
        cur_dilation = 1
        cur_padding = 0   
        self.conv31 = Conv2d(in_channels=cur_inchannels,
                            out_channels=cur_outchannels,
                            kernel_size=cur_kernel,
                            stride=cur_stride,
                            dilation=cur_dilation,
                            padding=cur_padding)
        self.conv31_act = ReLU()
        cur_height = ((cur_height + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1
        cur_width = ((cur_width + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1
        ### Convolution 3-2
        cur_inchannels = cur_outchannels
        cur_outchannels = 384
        cur_kernel = 3
        cur_stride = 1
        cur_dilation = 1
        cur_padding = 0   
        self.conv32 = Conv2d(in_channels=cur_inchannels,
                            out_channels=cur_outchannels,
                            kernel_size=cur_kernel,
                            stride=cur_stride,
                            dilation=cur_dilation,
                            padding=cur_padding)
        self.conv32_act = ReLU()
        cur_height = ((cur_height + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1
        cur_width = ((cur_width + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1
        ### Convolution 3-3
        cur_inchannels = cur_outchannels
        cur_outchannels = 256
        cur_kernel = 3
        cur_stride = 1
        cur_dilation = 1
        cur_padding = 0   
        self.conv33 = Conv2d(in_channels=cur_inchannels,
                            out_channels=cur_outchannels,
                            kernel_size=cur_kernel,
                            stride=cur_stride,
                            dilation=cur_dilation,
                            padding=cur_padding)
        self.conv33_act = ReLU()
        cur_height = ((cur_height + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1
        cur_width = ((cur_width + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1


        ### Pool 3
        cur_kernel = 3
        cur_stride = 2
        cur_dilation = 1
        cur_padding = 0 
        self.conv3_pool = MaxPool2d(kernel_size=cur_kernel,
                                    stride=cur_stride,
                                    dilation=cur_dilation,
                                    padding=cur_padding)
        self.conv3_lrn = LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        cur_height = ((cur_height + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1
        cur_width = ((cur_width + (2 * cur_padding) - (cur_dilation * (cur_kernel -1)) -1)//cur_stride)+1

        ### End of CNN layers, connect to Dense

        fc_in_features = self.channels * cur_height * cur_width

        # Fully Connected (Dense, Linear) 1
        self.fc1 = Linear(in_features=fc_in_features,out_features=4096)
        self.fc1_act = Tanh()
        self.fc1_drop = Dropout(p=0.5)

        # Fully Connected (Dense, Linear) 2
        self.fc2 = Linear(in_features=4096,out_features=4096)
        self.fc2_act = Tanh()
        self.fc2_drop = Dropout(p=0.5)

        # Fully Connected (Dense, Linear) 2
        self.out = Linear(in_features=4096,out_features=1)
        self.out_act = ReLU()

    def reshape_for_cnn(self,model_input):
        model_input = torch.reshape(model_input, (len(model_input),self.channels,self.height,self.width))
        # torch.Size([batch_size, channels, height, width])
        return model_input

    def cnn_layers(self,model_input):
        # Convolution 1
        output = self.conv1(model_input)
        output = self.conv1_act(output)
        output = self.conv1_pool(output)

        # Convolution 2
        output = self.conv2(output)
        output = self.conv2_act(output)
        output = self.conv2_pool(output)

        # Convolution 3
        output = self.conv31(output)
        output = self.conv31_act(output)
        output = self.conv32(output)
        output = self.conv32_act(output)
        output = self.conv33(output)
        output = self.conv33_act(output)
        output = self.conv3_pool(output)
        output = self.conv3_lrn(output)

        return output

    def flatten_conv(self,conv_output):
        output = conv_output.view(conv_output.size(0), -1)
        return output

    def dense_layers(self,conv_flat_output):

        # Fully Connected (Dense, Linear) 1
        output = self.fc1(conv_output)
        output = self.fc1_act(output)
        output = self.fc1_drop(output)

        # Fully Connected (Dense, Linear) 2
        output = self.fc2(output)
        output = self.fc2_act(output)
        output = self.fc2_drop(output)

        # Fully Connected (Dense, Linear) Output
        output = self.out(output)
        output = self.out_act(output)

    def forward(self,model_input):
        model_input = self.reshape_for_cnn(model_input)
        output = self.cnn_layers(model_input)
        output = self.flatten_conv(output)
        output = self.dense_layers(output)

        return output