import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.0)

    def forward(self, x):
        return self.dropout(x)

def resnet50_feature(pretrained=True):
    if pretrained:
        model = models.resnet50(pretrained=True)
        for parma in model.parameters():
            parma.requires_grad = True
    else:
        model = models.resnet50(pretrained=False)
    model.fc = Identity()
    return model
    