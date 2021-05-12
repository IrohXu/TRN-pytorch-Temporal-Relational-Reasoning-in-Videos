import torch.onnx
from torch.autograd import Variable
import onnx
from collections import OrderedDict
from model.TRN import TRN, MultiScaleTRN
from model.LRCN import LRCNs

num_frames = 8
num_segs = 8
num_class = 27
img_feature_dim = 2048

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MultiScaleTRN(img_feature_dim, num_frames, num_segs, num_class)
# model = LRCNs(img_feature_dim, 256, 1, num_class, DEVICE)

state_dict = torch.load('./log/best_model.pth')

model.load_state_dict(state_dict)
model_path = "best_model.onnx"
model.eval()

model.to(DEVICE)
dummy_input = Variable(torch.randn(1, 8, 3, 224, 224)).cuda()
torch.onnx.export(model, dummy_input, model_path, verbose=False)
