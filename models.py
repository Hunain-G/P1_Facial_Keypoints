## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel       
        ## input image size is 224 * 224 pixels        
        # first convolutional layer
        ## (W-F)/S + 1 = (224-5)/1 + 1 = 220
        ## self.conv1 = nn.Conv2d(1, 32, 5) # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5) # output tensor ((32, 220, 220))

        self.pool1 = nn.MaxPool2d(2,2) #output tensor (32,110, 110)
        
        self.conv2 = nn.Conv2d(32, 64, 5) # output tensor ((64, 106, 106))

        self.pool2 = nn.MaxPool2d(2,2) #output tensor (64, 53, 53)

        self.conv3 = nn.Conv2d(64, 128, 5) #output tensor (128, 49, 49)

        self.pool3 = nn.MaxPool2d(2,2) # output tensor (128, 24, 24)

        self.fc1 = nn.Linear(73728, 500)

        self.fc2 = nn.Linear(500, 500)

        self.fc3 = nn.Linear(500 , 136)

        self.drop1 = nn.Dropout(p = 0.1)

        self.drop2 = nn.Dropout(p = 0.2)

        self.drop3 = nn.Dropout(p = 0.3)


        # Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        # TODO: Define the feedforward behavior of this model
        # x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.relu(self.conv1(x)))
        
        #print ( "first layer" , x.shape)
        x = self.pool2(F.relu(self.conv2(x)))

        #print ( "second layer" , x.shape)

        x = self.pool3(F.relu(self.conv3(x)))

        #print ( "third layer" , x.shape)
        
        x = x.view(x.size(0), -1)
        #print(x.size)
        #print (x.shape)

        x = self.drop1(F.relu(self.fc1(x)))

        #print("first dense", x.shape)

        x = self.drop2(F.relu(self.fc2(x)))

        #print("second dense", x.shape)

        x = self.drop3(F.relu(self.fc3(x)))

        #print("third dense", x.shape)        

        # a modified x, having gone through all the layers of your model, should be returned
        return x
