from torch.utils.data import Dataset
from PIL import Image
import os
import os.path
import numpy as np
import torch

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class JesterDataset(Dataset):
    '''
    Generate Jester Dataset torch format
    '''
    def __init__(self, data_dir, annotation_dir, labels_dir, num_segs=2, transform=None):
        self.data_dir = data_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.num_segs = num_segs
        self.classes = list(np.loadtxt(labels_dir, dtype=np.str, delimiter=','))   # load jester-v1-labels.csv
        self.video_list = list()
        anno_array = np.loadtxt(annotation_dir, dtype=np.str, delimiter=';')
        for anno in anno_array:
            path = anno[0]
            label_name = anno[1]
            label = self.classes.index(label_name)
            data_path = os.path.join(data_dir, path)
            num_frames = len(os.listdir(data_path))
            self.video_list.append(VideoRecord([path, num_frames, label]))
    
    def _sample(self, num_total, num_segs):
        # sample = np.random.choice(range(num_total), size=num_segs, replace=None)
        # sample = np.sort(sample, axis=-1, kind='quicksort', order=None)
        sample = np.linspace(0, num_total-1, num_segs,endpoint=True,retstep=True,dtype=int)[0]
        return sample
    
    def __getitem__(self, index):
        assert index < len(self.video_list)
        info = self.video_list[index]
        target = info.label
        frames_num = info.num_frames
        data_path = os.path.join(self.data_dir, info.path)
        image_path_list = os.listdir(data_path)
        image_list = list()
        sample = self._sample(frames_num, self.num_segs)
        for i in sample:
            img_path = os.path.join(data_path, image_path_list[i])
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            image_list.append(img)
        image = torch.stack(image_list)
        return image, target

    def __len__(self):
        return len(self.video_list)