import os
import glob
import numpy as np
from PIL import Image
import skimage
import torch
from torch.utils.data import Dataset, DataLoader

def rgb_to_lab(image_rgb):
    
    image_rgb = skimage.transform.resize(image_rgb, (256, 256), anti_aliasing=True)    
    image_lab = skimage.color.rgb2lab(image_rgb)
    
    L = np.expand_dims(image_lab[:,:,0], 2)
    ab = image_lab[:,:,1:]
    
    L, ab = torch.from_numpy(L), torch.from_numpy(ab)
    L, ab = L.float(), ab.float()
    L, ab = L.permute(2, 0, 1), ab.permute(2, 0, 1)

    image_lab = torch.from_numpy(image_lab)
    image_lab = image_lab.float()
    image_lab = image_lab.permute(2, 0, 1)

    L = L / 50.0 - 1.0
    ab = ab / 110.0
    
    return image_lab, L, ab
    
def lab_to_rgb(L, ab):
    
    # L, ab = L.unsqueeze(0), ab.unsqueeze(0)
    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    L, ab = L.squeeze(0), ab.squeeze(0)
    L, ab = L.permute(1, 2, 0), ab.permute(1, 2, 0)
    L, ab = L.cpu().detach().numpy(), ab.cpu().detach().numpy()
    
    Lab = np.concatenate((L, ab), 2)
    rgb = skimage.color.lab2rgb(Lab)
    
    return rgb

class CustomImageDataset(Dataset):
    
    def __init__(self, folder_path):
        
        folder_path = os.path.normpath(folder_path)
        self.images_path = sorted(glob.glob(folder_path + '/*'))[:100]

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        
        image = Image.open(self.images_path[idx]).convert("RGB")
        image = np.asarray(image)
        
        return rgb_to_lab(image)

# dataset = CustomImageDataset('./dataset')
# data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# for i, im in enumerate(data_loader):
#     print(f'batch {i+1}: {type(im), len(im)}')
    
#     for idx in range(len(im)): 
#         print(f'{len(im[idx])} {im[idx].shape}')