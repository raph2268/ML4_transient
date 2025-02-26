import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, input_channel=1, n_outputs=2, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        
        # Adjusted input_channel to 1 for grayscale images
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1) # Output: (N, 128, 30, 30)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)           # Output: (N, 128, 30, 30)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)           # Output: (N, 128, 30, 30)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)           # Output: (N, 256, 15, 15) after first max pool
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)           # Output: (N, 256, 15, 15)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)           # Output: (N, 256, 15, 15)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)           # Output: (N, 512, 13, 13)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)           # Output: (N, 256, 11, 11)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)           # Output: (N, 128, 9, 9)

        self.l_c1 = nn.Linear(128, n_outputs)  # Linear layer for output classes
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)
        if self.top_bn:
            self.bn_c1 = nn.BatchNorm1d(n_outputs)  # For optional top batch normalization

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h = self.c2(h)
        h = F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = self.c3(h)
        h = F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)  # Reduces height and width by half
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)  # Further reduces height and width by half
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        h = F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h = self.c8(h)
        h = F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h = self.c9(h)
        h = F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        kernel_size = (h.size(2), h.size(3))  # Ensure kernel_size is a tuple of integers
        h = F.avg_pool2d(h, kernel_size)
        h = h.view(h.size(0), h.size(1))  # Flatten the tensor
        logit = self.l_c1(h)  # Linear layer
        if self.top_bn:
            logit = call_bn(self.bn_c1, logit)
        return logit
    
def call_bn(bn, x):
    return bn(x)


class CustomCNN(nn.Module):
    def __init__(self, input_shape, num_classes, filters_1, dropout_1, filters_2, dropout_2, units, dropout_3):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[2], out_channels=filters_1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(dropout_1)
        self.conv2 = nn.Conv2d(in_channels=filters_1, out_channels=filters_2, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(dropout_2)
        self.fc1 = nn.Linear(in_features=filters_2 * (input_shape[0] // 4) * (input_shape[1] // 4), out_features=units)
        self.dropout3 = nn.Dropout(dropout_3)
        self.fc2 = nn.Linear(in_features=units, out_features=1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x
