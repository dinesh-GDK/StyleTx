from dataloader import CustomImageDataset
from torch.utils.data import DataLoader
from unet import UNet
from train import train
from test import test_random_images

dataset = CustomImageDataset('./dataset')
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet(3, 1)

trained_model = train(model, data_loader, EPOCHS=20)
test_random_images(trained_model, './dataset', 3)
