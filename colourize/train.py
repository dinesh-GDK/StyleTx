import os
import csv
from tqdm import tqdm
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    monitor_loss = open(os.path.join(SAVE_PATH, "loss_metrics.csv"), "w+")
    csv_writer = csv.writer(monitor_loss)
    csv_writer.writerow(["S.No", "G Loss", "D Loss"])

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
        pbar = tqdm(total=len(data_loader), ncols=120, bar_format=b,
                    desc=f'Epochs: {epoch+1}/{EPOCHS}')

        running_loss = {
            "generator": 0.0,
            "discriminator": 0.0
        }

        for data in data_loader:
            
            l_L, l_ab = data
            l_L, l_ab = l_L.to(DEVICE), l_ab.to(DEVICE)
            
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            l_g_output = generator(l_L)
            
            # discriminator
            l_d_g_image = discriminator(l_g_output.detach())

            l_D_LABEL_TRUE = torch.Tensor([1.0]).expand(l_d_g_image.shape).to(DEVICE)
            l_D_LABEL_FALSE = torch.Tensor([0.0]).expand(l_d_g_image.shape).to(DEVICE)
            
            l_d_loss1 = criterion1(l_d_g_image, l_D_LABEL_FALSE)

            l_d_ab = discriminator(l_ab)
            l_d_loss2 = criterion1(l_d_ab, l_D_LABEL_TRUE)
            l_d_loss = (l_d_loss1 + l_d_loss2) * 0.5
            l_d_loss.backward()
            d_optimizer.step()        
            
            # generator
            d_g_image = discriminator(l_g_output)
            l_g_loss1 = criterion1(d_g_image, l_D_LABEL_TRUE)
            l_g_loss2 = criterion2(l_g_output, l_ab) * LAMBDA
            l_g_loss = l_g_loss1 + l_g_loss2
            l_g_loss.backward()
            g_optimizer.step()

            running_loss["generator"] += float(l_g_loss / len(data))
            running_loss["discriminator"] += float(l_d_loss / len(data))

            pbar.update(1)
            pbar.set_postfix(G_Loss=f'{(l_g_loss/len(data)):.4f}', D_Loss=f'{(l_d_loss/len(data)):.4f}')

            # for variable in dir():
            #     if variable.startswith("l_"):
            #         del locals()[variable]

        pbar.close()

        running_loss["generator"] /= len(data_loader)
        running_loss["discriminator"] /= len(data_loader)

        csv_writer.writerow([epoch+1, f"{running_loss['generator']:.4f}", f"{running_loss['generator']:.4f}"])

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
            }, os.path.join(SAVE_PATH, "model.pt"))

            EPOCH_LOSS = running_loss["generator"]

    monitor_loss.close()

    return generator
