from torch.utils.data import dataloader
from tqdm import tqdm
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_LABEL_TRUE, D_LABEL_FALSE = None, None

def load_model(PATH, generator, discriminator, g_optimizer, d_optimizer):

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

def set_labels(shape):
    
    global D_LABEL_TRUE, D_LABEL_FALSE
    
    D_LABEL_TRUE, D_LABEL_FALSE = torch.Tensor([1.0]), torch.Tensor([0.0])
    D_LABEL_TRUE = D_LABEL_TRUE.expand(shape)
    D_LABEL_FALSE = D_LABEL_FALSE.expand(shape)    
    D_LABEL_TRUE = D_LABEL_TRUE.to(DEVICE)
    D_LABEL_FALSE = D_LABEL_FALSE.to(DEVICE)

def train(generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    criterion1,
    criterion2,
    data_loader,
    EPOCHS,
    SAVE_PATH,
    LOAD_PATH,
    LAMBDA=100.0):

    print('Training in', DEVICE, '...')

    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)

    if LOAD_PATH is not None:
        load_model(LOAD_PATH, generator, discriminator, g_optimizer, d_optimizer)

    generator = generator.train()
    discriminator = discriminator.train()

    EPOCH_LOSS = float("inf")

    for epoch in range(EPOCHS):
    
        b='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        pbar = tqdm(total=len(data_loader), ncols=100, bar_format=b,
                    desc=f'Epochs: {epoch+1}/{EPOCHS}')

        running_loss = {
            "generator": 0.0,
            "discriminator": 0.0
        }
        
        for data in data_loader:
            
            L, ab = data
            L, ab = L.to(DEVICE), ab.to(DEVICE)
            
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            g_output = generator(L)
            
            # discriminator
            d_g_image = discriminator(g_output.detach())
            if D_LABEL_TRUE is None: set_labels(d_g_image.shape)
            d_loss1 = criterion1(d_g_image, D_LABEL_FALSE)
            d_ab = discriminator(ab)
            d_loss2 = criterion1(d_ab, D_LABEL_TRUE)
            d_loss = (d_loss1 + d_loss2) * 0.5
            d_loss.backward()
            d_optimizer.step()        
            
            # generator
            d_g_image = discriminator(g_output)
            g_loss1 = criterion1(d_g_image, D_LABEL_TRUE)
            g_loss2 = criterion2(g_output, ab) * LAMBDA
            g_loss = g_loss1 + g_loss2
            g_loss.backward()
            g_optimizer.step()

            running_loss["generator"] += g_loss.item() / len(data)
            running_loss["discriminator"] += d_loss.item() / len(data)

            pbar.update(1)
            pbar.set_postfix(G_Loss=f'{(g_loss.item()/len(data)):.4f}', D_Loss=f'{(d_loss.item()/len(data)):.4f}')

        pbar.close()

        running_loss["generator"] /= len(data_loader)
        running_loss["discriminator"] /= len(data_loader)

        if running_loss["generator"] < EPOCH_LOSS:
            print(f"New best model: Loss imporved from {EPOCH_LOSS:0.4f} to {running_loss['generator']:.4f}")

            torch.save({
                "epoch": epoch,
                "model": {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict()
                },
                "optimizer": {
                    "generator": g_optimizer.state_dict(),
                    "discriminator": d_optimizer.state_dict()
                },
                "loss": running_loss
            }, SAVE_PATH)

            EPOCH_LOSS = running_loss["generator"]

    return generator
