import glob
import random
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

def test_random_images(model, folder_path, samples):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Testing in', device, '...')

    model.eval()
    
    image_paths = glob.glob(folder_path + '/*')
    
    for i in range(samples):
        
        idx = random.randint(0, len(image_paths)-1)
        
        image = Image.open(image_paths[idx])
        image = image.resize((256, 256))
        image = np.asarray(image) / 255.0
        
        image_torch = torch.from_numpy(image)
        image_torch = image_torch.permute(2, 0, 1)
        image_torch = torch.unsqueeze(image_torch, 0).float()
        image_torch = image_torch.to(device)
        
        out_torch = model(image_torch)
        
        out_torch = out_torch.squeeze(0).squeeze(0)
        out = out_torch.cpu().detach().numpy()

        fig, ax = plt.subplots(1, 2)
        fig.suptitle(f'Sample: {i+1}')
        ax[0].imshow(image)
        ax[1].imshow(out, cmap='gray')        
        plt.show()
