import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .resnet50 import resnet50_feature
from .resnet import resnet50
from .TRNmodule import MergeFrame, RelationModule, RelationModuleMultiSample, RelationModuleMultiScale

class TRN(nn.Module):
    def __init__(self, img_feature_dim, num_frames, num_segs, num_class):
        super(TRN, self).__init__()
        self.num_class = num_class
        self.num_frames = num_frames
        self.num_segs = num_segs
        self.img_feature_dim = img_feature_dim
        self.backbone = resnet50(pretrained=True)
        self.merge = MergeFrame()
        self.TRN = RelationModule(img_feature_dim, num_frames, num_class)

    def forward(self, x):
        total_frame = x.shape[1]
        x_frame = torch.chunk(x, total_frame, 1)
        sample = np.random.choice(range(total_frame), size=self.num_frames, replace=None)
        sample = np.sort(sample, axis=-1, kind='quicksort', order=None)
        x_feature = list()
        for i in sample:
            x_feature.append(torch.unsqueeze(self.backbone(torch.squeeze(x_frame[i], 1)), 1))
        x = self.merge(x_feature)
        x = self.TRN(x)
        return x

class MultiScaleTRN(nn.Module):
    def __init__(self, img_feature_dim, num_frames, num_segs, num_class):
        super(MultiScaleTRN, self).__init__()
        self.num_class = num_class
        self.num_frames = num_frames
        self.img_feature_dim = img_feature_dim
        self.backbone = resnet50(pretrained=True)
        self.merge = MergeFrame()
        self.TRN = RelationModuleMultiScale(img_feature_dim, num_frames, num_segs, num_class)
        # self.TRN = RelationModuleMultiSample(img_feature_dim, num_frames, num_segs, num_class)
    
    def forward(self, x):
        total_frame = x.shape[1]
        x_frame = torch.chunk(x, total_frame, 1)
        x_feature = list()
        for i in range(0, total_frame):
            x_feature.append(torch.unsqueeze(self.backbone(torch.squeeze(x_frame[i], 1)), 1))
        x = self.merge(x_feature)
        x = self.TRN(x)
        return x

if __name__ == "__main__":
    batch_size = 5
    num_frames = 1
    num_segs = 8
    num_class = 27
    img_feature_dim = 2048
    input_var = Variable(torch.randn(batch_size, num_segs, 3, 224, 224))
    # model = MultiScaleTRN(img_feature_dim, num_frames, num_segs, num_class)
    model = TRN(img_feature_dim, num_frames, num_segs, num_class)
    output = model(input_var)
    print(output)
    print(output.shape)