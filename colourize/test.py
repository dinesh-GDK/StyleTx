import glob
import random
import skimage
import matplotlib.pyplot as plt
import torch

from dataloader import rgb_to_lab, lab_to_rgb

def test_random_images(model, folder_path, samples):

    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('Testing in', device, '...')

    model.eval()
    
    image_paths = glob.glob(folder_path + '/*')
    
    for i in range(samples):
        
        idx = random.randint(0, len(image_paths)-1)
        
        image = skimage.io.imread(image_paths[idx])
        
        L, _ = rgb_to_lab(image)
        
        L = L.unsqueeze(0).to(device)
        
        out = model(L)
        
        rgb = lab_to_rgb(L, out)

        fig, ax = plt.subplots(1, 2)
        fig.suptitle(f'Sample: {i+1}')
        ax[0].imshow(image)
        ax[1].imshow(rgb)        
        plt.show()
