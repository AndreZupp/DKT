from models.resnet import resnet18
from models.resnet32 import resnet32
from models.from_avalanche import SlimResNet18
from torch import nn

    
class FakeModel(nn.Module):
    def __init__(self, model, output_layer):
        super().__init__()
        self.output_layer = output_layer
        self.pretrained = model
        self.children_list = []
        for n,c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == self.output_layer:
                break

        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None
        
    def forward(self,x):
        x = self.net(x)
        return x