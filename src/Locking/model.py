"""
    Sample Model architecutre. We can modify it based on our needs as we start training
"""

import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, Conv2d, BatchNorm2d, Dropout, MaxPool2d
from torchsummary import summary


class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 512, kernel_size=3, stride=2),
            Dropout(0.6, inplace= True),
            BatchNorm2d(512),
            ReLU(inplace=True),

            #MaxPool2d(kernel_size=2, stride=2),

            Conv2d(512, 256, kernel_size=3, stride=2),
            Dropout(0.6, inplace= True),
            BatchNorm2d(256),
            ReLU(inplace=True),

            # Conv2d(256, 128, kernel_size=3, stride=2),
            # Dropout(0.6, inplace= True),
            # BatchNorm2d(128),
            # ReLU(),

            #MaxPool2d(kernel_size=2, stride=2),
        )

        # flatten the volume, so we need height and width.also, dont forget the maxpool
        self.linear = Sequential(
            Linear(209664, 512), # the input for  the linear layer coming from a conv one 
            Dropout(0.6, inplace= True),

            Linear(512, 256),
            Dropout(0.6, inplace= True),

            Linear(256, 128),
            Dropout(0.6, inplace= True),

            Linear(128, 64),
            Dropout(0.6, inplace= True),

            Linear(64, 3)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Current device: ", device)

model = CNN()
# model = model.cuda()
model = model.to('cuda:0') # send the model to device cause sometimes there are errors regarding device

# print(summary(model, (1,  90,160)))