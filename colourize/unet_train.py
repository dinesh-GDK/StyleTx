import os
from datetime import datetime
import csv
from tqdm import tqdm
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unet_train(generator,
    optimizer,
    criterion,
    data_loader,
    EPOCHS,
    SAVE_PATH):

    monitor_loss = open(os.path.join(SAVE_PATH, "loss_metrics.csv"), "w+")
    csv_writer = csv.writer(monitor_loss)
    csv_writer.writerow(["Epoch", "TIME", "G Loss"])
    monitor_loss.close()

    print('Training in', DEVICE, '...')

    generator = generator.train()

    EPOCH_LOSS = {
        "generator": float("inf"),      
    }

    for epoch in range(EPOCHS):
    
        b='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        pbar = tqdm(total=len(data_loader), ncols=120, bar_format=b,
                    desc=f'Epochs: {epoch+1}/{EPOCHS}')

        running_loss = {
            "generator": 0.0,
        }

        for data in data_loader:
            
            l_L, l_ab = data
            l_L, l_ab = l_L.to(DEVICE), l_ab.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = generator(l_L)
            
            loss = criterion(outputs, l_ab)
            loss.backward()
            optimizer.step()

            running_loss["generator"] += float(loss / len(data))

            pbar.update(1)
            pbar.set_postfix(G_Loss=f'{(loss/len(data)):.4f}')

        pbar.close()

        running_loss["generator"] /= len(data_loader)

        if running_loss["generator"] < EPOCH_LOSS["generator"]:
            
            print(f"New best model: Generator loss imporved from {EPOCH_LOSS['generator']:0.4f} to {running_loss['generator']:.4f}")

            torch.save({
                "epoch": epoch+1,
                "model": {
                    "generator": generator.state_dict(),
                },
                "optimizer": {
                    "generator": optimizer.state_dict(),
                },
                "loss": running_loss
            }, os.path.join(SAVE_PATH, "model.pt"))

            EPOCH_LOSS = running_loss
        else:
            print(f"Loss did not improve; Current Loss: {running_loss['generator']:.4f}; Best Loss: {EPOCH_LOSS['generator']:0.4f}")

        monitor_loss = open(os.path.join(SAVE_PATH, "loss_metrics.csv"), "a+")
        csv_writer = csv.writer(monitor_loss)
        csv_writer.writerow([epoch+1, datetime.now().strftime("%Y%m%d-%H%M%S"), f"{running_loss['generator']:.4f}"])
        monitor_loss.close()

    return generator