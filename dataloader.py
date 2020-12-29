import torchvision.transforms as tfs
import os
from PIL import Image
import numpy as np
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self,path_root="./dataset/"):
        super(Dataset,self).__init__()
        self.cloud_images_dir=os.listdir(os.path.join(path_root,"cloud"))
        self.cloud_images=[os.path.join(path_root,"cloud",img) for img in self.cloud_images_dir]
        self.gt_images_dir=os.listdir(os.path.join(path_root,"label"))
        self.gt_images=[os.path.join(path_root,"label",img) for img in self.gt_images_dir]

    def __getitem__(self, item):
        cloud=Image.open(self.cloud_images[item])
        gt=Image.open(self.gt_images[item])

        cloud=tfs.ToTensor()(cloud)
        gt=tfs.ToTensor()(gt)

        return cloud,gt

    def __len__(self):
        return len(self.cloud_images)
