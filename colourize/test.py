import glob
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

from dataloader import rgb_to_lab, lab_to_rgb

def test_random_images(model, folder_path, samples):

    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Testing in', device, '...')

    model.eval()
    
    image_paths = glob.glob(folder_path + '/*')
    
    for i in range(samples):
        
        idx = random.randint(0, len(image_paths)-1)
        
        image = Image.open(image_paths[idx]).convert("RGB")
        image = np.asarray(image)
        
        _, L, _ = rgb_to_lab(image)
        
        L = L.unsqueeze(0).to(device)
        
        out = model(L)
        
        rgb = lab_to_rgb(L, out)

        fig, ax = plt.subplots(1, 3)
        fig.suptitle(f'Sample: {i+1}')
        ax[0].imshow(image)
        ax[1].imshow(image[:,:,0], cmap="gray")
        ax[2].imshow(rgb)        
        plt.show()
        # fig.savefig(f"{i}.png")
