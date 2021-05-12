import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from model.TRN import TRN, MultiScaleTRN
from model.LRCN import LRCNs
from utils.dataset import JesterDataset

DATA_DIR = './log/100013'
LABELS_DIR = './label/jester-v1-labels.csv'

num_frames = 8
num_segs = 8
num_class = 27
img_feature_dim = 2048

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MultiScaleTRN(img_feature_dim, num_frames, num_segs, num_class)
# model = LRCNs(img_feature_dim, 256, 1, num_class, DEVICE)

state_dict = torch.load('./log/best_model.pth')

model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_test_data(data_path):
    image_path_list = os.listdir(data_path)
    frames_num = len(image_path_list)
    image_list = list()
    sample = np.linspace(0, frames_num-1, num_segs, endpoint=True,retstep=True,dtype=int)[0]
    for i in sample:
        img_path = os.path.join(data_path, image_path_list[i])
        img = Image.open(img_path)
        img = transform(img)
        image_list.append(img)
    image = torch.stack(image_list)
    return image


if __name__ == "__main__":
    dummy_input = load_test_data(DATA_DIR)
    classes = list(np.loadtxt(LABELS_DIR, dtype=np.str, delimiter=','))
    dummy_input = torch.unsqueeze(dummy_input, dim=0).cuda()
    outputs = model(dummy_input)
    _, preds = torch.max(outputs, 1)

    print(classes[int(preds.to('cpu'))])

