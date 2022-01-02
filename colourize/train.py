from tqdm import tqdm
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_LABEL_TRUE, D_LABEL_FALSE = None, None

def set_labels(shape):
    
    global D_LABEL_TRUE, D_LABEL_FALSE
    
    D_LABEL_TRUE, D_LABEL_FALSE = torch.Tensor([1.0]), torch.Tensor([0.0])
    D_LABEL_TRUE = D_LABEL_TRUE.expand(shape)
    D_LABEL_FALSE = D_LABEL_FALSE.expand(shape)    
    D_LABEL_TRUE = D_LABEL_TRUE.to(DEVICE)
    D_LABEL_FALSE = D_LABEL_FALSE.to(DEVICE)

def train(generator, discriminator, data_loader, EPOCHS=200):

    print('Training in', DEVICE, '...')

    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()

    for epoch in range(EPOCHS):
    
        b='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        pbar = tqdm(total=len(data_loader), ncols=100, bar_format=b,
                    desc=f'Epochs: {epoch+1}/{EPOCHS}')
        
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
            g_loss2 = criterion2(g_output, ab) * 100.0
            g_loss = g_loss1 + g_loss2
            g_loss.backward()
            g_optimizer.step()
            
            pbar.update(1)
            pbar.set_postfix(Loss=f'{(g_loss.item()/len(data)):.4f}')
        pbar.close()

    return generator
