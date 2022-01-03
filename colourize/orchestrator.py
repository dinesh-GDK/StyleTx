import os
from datetime import datetime
from dataloader import CustomImageDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet import UNet
from classifier import PatchClassifier
from train import train
from test import test_random_images

dataset = CustomImageDataset('./dataset')
data_loader = DataLoader(dataset, batch_size=6, shuffle=True)

generator = UNet(1, 2)
discriminator = PatchClassifier(2, 1)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss()

LOAD_PATH = None
# LOAD_PATH = "./RESULTS/20220102-185849.pt"

SAVE_PATH = "./RESULTS/" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pt"
os.makedirs("./RESULTS", exist_ok=True)

trained_model = train(generator,
                    discriminator,
                    g_optimizer,
                    d_optimizer,
                    criterion1,
                    criterion2,
                    data_loader,
                    EPOCHS=20,
                    SAVE_PATH=SAVE_PATH,
                    LOAD_PATH=LOAD_PATH)

test_random_images(trained_model, './dataset', 3)
