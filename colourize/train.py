import os
from datetime import datetime
import csv
from tqdm import tqdm
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    criterion1,
    criterion2,
    data_loader,
    EPOCHS,
    SAVE_PATH,
    LAMBDA):

    monitor_loss = open(os.path.join(SAVE_PATH, "loss_metrics.csv"), "w+")
    csv_writer = csv.writer(monitor_loss)
    csv_writer.writerow(["Epoch", "TIME", "G Loss", "D Loss"])
    monitor_loss.close()

    print('Training in', DEVICE, '...')

    generator = generator.train()
    discriminator = discriminator.train()

    EPOCH_LOSS = {
        "generator": float("inf"),    
        "discriminator": float("inf")    
    }

    for epoch in range(EPOCHS):
    
        b='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        pbar = tqdm(total=len(data_loader), ncols=120, bar_format=b,
                    desc=f'Epochs: {epoch+1}/{EPOCHS}')

        running_loss = {
            "generator": 0.0,
            "discriminator": 0.0
        }

        for data in data_loader:
            
            Lab, L, ab = data
            Lab, L, ab = Lab.to(DEVICE), L.to(DEVICE), ab.to(DEVICE)
            
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            g_output = generator(L)
            g_output = torch.cat([L, g_output], dim=1)
            
            # discriminator
            d_g_image = discriminator(g_output.detach())

            LABEL_TRUE = torch.Tensor([1.0]).expand(d_g_image.shape).to(DEVICE)
            LABEL_FALSE = torch.Tensor([0.0]).expand(d_g_image.shape).to(DEVICE)
            
            d_loss1 = criterion1(d_g_image, LABEL_FALSE)

            d_ab = discriminator(Lab)
            d_loss2 = criterion1(d_ab, LABEL_TRUE)
            d_loss = (d_loss1 + d_loss2) * 0.5
            d_loss.backward()
            d_optimizer.step()        
            
            # generator
            d_g_image = discriminator(g_output)
            g_loss1 = criterion1(d_g_image, LABEL_TRUE)
            g_loss2 = criterion2(g_output, Lab) * LAMBDA
            g_loss = g_loss1 + g_loss2
            g_loss.backward()
            g_optimizer.step()

            running_loss["generator"] += float(g_loss / len(data))
            running_loss["discriminator"] += float(d_loss / len(data))

            pbar.update(1)
            pbar.set_postfix(G_Loss=f'{(g_loss/len(data)):.4f}', D_Loss=f'{(d_loss/len(data)):.4f}')

            # for variable in dir():
            #     if variable.startswith("l_"):
            #         del locals()[variable]

        pbar.close()

        running_loss["generator"] /= len(data_loader)
        running_loss["discriminator"] /= len(data_loader)

        if running_loss["generator"] < EPOCH_LOSS["generator"] or running_loss["discriminator"] < EPOCH_LOSS["discriminator"]:
            
            if running_loss["generator"] < EPOCH_LOSS["generator"]:
                print(f"New best model: Generator loss imporved from {EPOCH_LOSS['generator']:0.4f} to {running_loss['generator']:.4f}")
            else:
                print(f"New best model: Discriminator loss imporved from {EPOCH_LOSS['discriminator']:0.4f} to {running_loss['discriminator']:.4f}")

            torch.save({
                "epoch": epoch+1,
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

            EPOCH_LOSS = running_loss
        else:
            print(f"Loss did not improve; Current Loss: {running_loss['generator']:.4f}; Best Loss: {EPOCH_LOSS['generator']:0.4f}")

        monitor_loss = open(os.path.join(SAVE_PATH, "loss_metrics.csv"), "a+")
        csv_writer = csv.writer(monitor_loss)
        csv_writer.writerow([epoch+1, datetime.now().strftime("%Y%m%d-%H%M%S"), f"{running_loss['generator']:.4f}", f"{running_loss['discriminator']:.4f}"])
        monitor_loss.close()

    return generator
