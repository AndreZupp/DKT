from avalanche.models.slim_resnet18 import ResNet, SlimResNet18
from models.resnet32 import resnet32
from models.resnet import resnet18, resnet34
import torch.nn as nn


class CustomSlimResnet18(nn.Module):

    def __init__(self, number_of_classes=100, nf=20):
        super(CustomSlimResnet18, self).__init__()
        self.slim_res18 = SlimResNet18(number_of_classes, nf=nf)
        # Changing the last layer which was used for classification, now it is transparent wrt its input
        resnet18_output_size = self.slim_res18.linear.in_features
        self.slim_res18.linear = nn.Identity()
        self.cl = nn.Linear(resnet18_output_size, number_of_classes)
        self.student = nn.Linear(resnet18_output_size, number_of_classes)

    def forward(self, x):
        features = self.slim_res18(x)
        cl_out = self.cl(features)
        student_out = self.student(features)
        return cl_out, student_out

class CustomResnet32(nn.Module):
    def __init__(self, number_of_classes=100, nf=20):
        super(CustomResnet32, self).__init__()
        self.res32 = resnet32()
        # Changing the last layer which was used for classification, now it is transparent wrt its input
        resnet18_output_size = self.res32.linear.in_features
        self.res32.linear = nn.Identity()
        self.cl = nn.Linear(resnet18_output_size, number_of_classes)
        self.student = nn.Linear(resnet18_output_size, number_of_classes)

    def forward(self, x):
        features = self.res32(x)
        cl_out = self.cl(features)
        student_out = self.student(features)
        return cl_out, student_out
    
class CustomResnet18(nn.Module):

    def __init__(self, number_of_classes=100, nf=20):
        super(CustomSlimResnet18, self).__init__()
        self.res18 = resnet18()
        # Changing the last layer which was used for classification, now it is transparent wrt its input
        resnet18_output_size = self.slim_res18.fc.in_features
        self.res18.fc = nn.Identity()
        self.cl = nn.Linear(resnet18_output_size, number_of_classes)
        self.student = nn.Linear(resnet18_output_size, number_of_classes)

    def forward(self, x):
        features = self.res18(x)
        cl_out = self.cl(features)
        student_out = self.student(features)
        return cl_out, student_out

class CustomResnet34(nn.Module):

    def __init__(self, number_of_classes=100):
        super(CustomResnet34, self).__init__()
        self.res34 = resnet34()
        output_size = self.res34.fc.in_features
        self.res34.fc = nn.Identity()
        self.cl = nn.Linear(output_size, number_of_classes)
        self.student = nn.Linear(output_size, number_of_classes)

    def forward(self, x):
        features = self.res34(x)
        cl_out = self.cl(features)
        student_out = self.student(features)
        return cl_out, student_out