import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
# from torch.utils.data import DataLoader

class CustomImageDataset(Dataset):
    
    def __init__(self, folder_path):
        
        folder_path = os.path.normpath(folder_path)
        self.images_path = glob.glob(folder_path + '/*')

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        
        image = Image.open(self.images_path[idx])
        label = image.copy().convert('L')
        
        image, label = self.prepare(image), self.prepare(label, True)
        
        return [image, label]
        
    def prepare(self, image, add_dim=False):
        
        image = np.asarray(image)
        
        if add_dim:
            image = np.expand_dims(image, 2)
        
        image = image / 255.0
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        image = F.resize(image, size=[256, 256])
        
        return image

# dataset = CustomImageDataset('./dataset')
# data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# for i, im in enumerate(data_loader):
#     print(f'batch {i+1}: {type(im), len(im)}')
    
#     for idx in range(len(im)): 
#         print(f'{len(im[idx])} {im[idx].shape}')