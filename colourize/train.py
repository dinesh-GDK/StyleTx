from tqdm import tqdm
import torch
import torch.nn as nn

def train(model, data_loader, EPOCHS=200):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Training in', device, '...')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        
        b='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        pbar = tqdm(total=len(data_loader), bar_format=b,
                    desc=f'Epochs: {epoch+1}/{EPOCHS}')

        for data in data_loader:
            
            inputs, labels = data
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            pbar.update(1)
            pbar.set_postfix(Loss=f'{(loss/len(data)):.4f}')
        pbar.close()

    return model