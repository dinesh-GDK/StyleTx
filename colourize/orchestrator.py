from dataloader import CustomImageDataset
from torch.utils.data import DataLoader
from unet import UNet
from classifier import PatchClassifier
from train import train
from test import test_random_images

dataset = CustomImageDataset('./dataset')
data_loader = DataLoader(dataset, batch_size=6, shuffle=True)

generator = UNet(1, 2)
discriminator = PatchClassifier(2, 1)

trained_model = train(generator, discriminator, data_loader, EPOCHS=500)
test_random_images(trained_model, './dataset', 3)
