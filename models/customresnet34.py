from models.resnet import resnet34
import torch.nn as nn


class CustomResnet34(nn.Module):

    def __init__(self, number_of_classes=100):
        super(CustomResnet34, self).__init__()
        self.resnet_model = resnet34()
        output_size = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Identity()
        self.cl = nn.Linear(output_size, number_of_classes)
        self.student = nn.Linear(output_size, number_of_classes)

    def forward(self, x):
        features = self.resnet_model(x)
        cl_out = self.cl(features)
        student_out = self.student(features)
        return cl_out, student_out
