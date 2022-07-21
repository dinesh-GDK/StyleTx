import os
from datetime import datetime
from dataloader import CustomImageDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet import UNet
from classifier import PatchClassifier
from train import train
from unet_train import unet_train
from test import test_random_images

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

def load_model(PATH, generator, discriminator, g_optimizer, d_optimizer):
# def load_model(PATH, generator, g_optimizer):

    cp = torch.load(PATH, map_location=DEVICE)

    print(f"Loading model saved at\n \
            EPOCH: {cp['epoch']}\n \
            G_LOSS: {cp['loss']['generator']:.4f}\n \
            D_LOSS: {cp['loss']['discriminator']:.4f}")

    generator.load_state_dict(cp["model"]["generator"])
    discriminator.load_state_dict(cp["model"]["discriminator"])

    g_optimizer.load_state_dict(cp["optimizer"]["generator"])
    d_optimizer.load_state_dict(cp["optimizer"]["discriminator"])

    del cp

# dataset = CustomImageDataset('./dataset')
dataset = CustomImageDataset('/mnt/c/Users/dines/projects/coco_sample/train_sample')
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

generator = UNet(1, 2)
discriminator = PatchClassifier(3, 1)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

# criterion1 = nn.MSELoss()
criterion1 = nn.BCEWithLogitsLoss()
criterion2 = nn.L1Loss()

generator = generator.to(DEVICE)
discriminator = discriminator.to(DEVICE)

SAVE_PATH = "./RESULTS/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(SAVE_PATH, exist_ok=True)

trained_model = train(generator,
                    discriminator,
                    g_optimizer,
                    d_optimizer,
                    criterion1,
                    criterion2,
                    data_loader,
                    EPOCHS=100,
                    SAVE_PATH=SAVE_PATH,
                    LAMBDA=20.0)

# LOAD_PATH = "/mnt/c/Users/dines/projects/StyleTx/colourize/RESULTS/20220104-143334/model.pt"
# # # LOAD_PATH = "/mnt/c/Users/dines/projects/StyleTx/RESULTS/20220104-005517/model.pt"
# load_model(LOAD_PATH, generator, discriminator, g_optimizer, d_optimizer)
# # # load_model(LOAD_PATH, generator, g_optimizer)
# test_random_images(generator, '/mnt/c/Users/dines/projects/coco_sample/train_sample', 10)
# # test_random_images(generator, '/mnt/c/Users/dines/projects/StyleTx/colourize/dataset', 3)







# unet_train(generator,
#     g_optimizer,
#     criterion1,
#     data_loader,
#     EPOCHS=100,
#     SAVE_PATH=SAVE_PATH)