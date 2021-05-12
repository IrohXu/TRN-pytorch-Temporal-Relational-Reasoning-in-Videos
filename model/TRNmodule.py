import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class MergeFrame(nn.Module):
    def __init__(self):
        super(MergeFrame, self).__init__()
    
    def forward(self, x):
        if(len(x)==1):
            return x[0]
        return torch.cat(x, dim=1)

class RelationModule(nn.Module):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.classifier = self.fc_fusion()
    
    def fc_fusion(self):
        # naive concatenate
        num_bottleneck = 256    # This part is chosen by the author.
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck, self.num_class),
                )
        return classifier
    
    def forward(self, input):
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        input = self.classifier(input)
        return input

class RelationModuleMultiSample(nn.Module):
    
    def __init__(self, img_feature_dim, num_frames, num_segs, num_class):
        super(RelationModuleMultiSample, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.num_segs = num_segs
        self.num_subsample = 1    # The parameter k mentioned in the paper.
        self.classifier = self.fc_fusion()
        self.relations_scale = self.return_relationset(num_segs, num_frames)
    
    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

    def fc_fusion(self):
        # naive concatenate
        num_bottleneck = 256    # This part is chosen by the author.
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck, self.num_class),
                )
        return classifier
    
    def forward(self, input):
        idx_relations_randomsample = np.random.choice(len(self.relations_scale), self.num_subsample, replace=False)
        act_all = input[:, self.relations_scale[idx_relations_randomsample[0]] , :]
        act_all = act_all.view(act_all.size(0), self.num_frames * self.img_feature_dim)
        act_all = self.classifier(act_all)
        for i in range(1, self.num_subsample):
            act_relation = input[:, self.relations_scale[idx_relations_randomsample[i]], :]
            act_relation = act_relation.view(act_relation.size(0), self.num_frames * self.img_feature_dim)
            act_relation = self.classifier(act_relation)
            act_all += act_relation
        return act_all


class RelationModuleMultiScale(nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_frames, num_segs, num_class):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3   # how many relations selected to sum up
        self.num_segs = num_segs
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_segs, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Linear(num_bottleneck, self.num_class),
                        )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


if __name__ == "__main__":
    batch_size = 10
    num_frames = 8
    num_segs = 30
    num_class = 27
    img_feature_dim = 512
    input_var = Variable(torch.randn(batch_size, num_segs, img_feature_dim))
    # model = RelationModuleMultiScale(img_feature_dim, num_frames, num_segs, num_class)
    model = RelationModuleMultiSample(img_feature_dim, num_frames, num_segs, num_class)
    output = model(input_var)
    print(output)