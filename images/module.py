
#def default_transform(image):


import glob, os
import numpy as np
import cv2
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, image_dir, label_from_filename, size, interpolation=cv2.INTER_AREA, x_keyname='image', y_keyname='label', transform=None):
        assert len(size) == 2
        self.files = glob.glob(os.path.join(image_dir, '**'))
        self.files.sort()
        self.label_from_filename = label_from_filename
        self.label_set = self.compute_labels()
        self.transform = transform
        self.size = size
        self.interpolation = interpolation
        self.x_keyname = x_keyname
        self.y_keyname = y_keyname

    def check(self):
        assert all(map(lambda fn:os.path.exists(fn), self.files))
    
    def compute_labels(self):
        self.labels = list(map(self.label_from_filename, self.files))
        return set(self.labels)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        label = self.labels[idx]
        image = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size, self.interpolation)#.transpose(2,0,1).astype(np.float32)
        image = self.transform(image) if self.transform else image
        return { self.x_keyname:image, self.y_keyname:label }


