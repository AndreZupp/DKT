from models.resnet32 import resnet32
import torch.nn as nn
from avalanche.models import IncrementalClassifier
from torch.nn import Linear

 
class MultiHeadResnet(nn.Module):

    def __init__(self, class_per_task=10):
        super(MultiHeadResnet, self).__init__()
        self.resnet_model = resnet32()
        output_size = self.resnet_model.linear.in_features
        self.resnet_model.linear = nn.Identity()
        self.classifier = IncrementalClassifier(output_size, class_per_task)

    def forward(self, x):
        features = self.resnet_model(x)
        out = self.classifier(features)
        return out

    def adaptation(self, experience):
        self.classifier.adaptation(experience)
