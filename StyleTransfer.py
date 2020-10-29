import torch
import numpy as np
from torchvision import models
from PIL import Image
import process

import tkinter as tk
from tkinter import ttk

def StyleTransfer(content_img, style_img, universal, alpha=1, beta=10, epochs=500):
    ''' Transfers the style of the image given by the user
    INPUT:
        content_img - the image for which the style transfer is to be applied (type: PIL)
        style_img - the image whose style is to be transfered to content_img (type: PIL)
        universal - info of background & button colours and icon path (type: dictionary)
        alpha - weight of the content info used in the error function (type: positive int)
        beta -  weigth of the style info used in the error function (type: positive int)
        epochs - number of iterations to be done (type: positive int)
        checkpoint - display the intermediate result for every checkpoint (type: positive int
    OUTPUT:
        target_img - style transfered image (type: PIL)
    '''

    # set device to train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # creating window to display progress bar
    mini_frame = tk.Tk()
    mini_frame.geometry('300x150')
    mini_frame.iconbitmap(universal['icon_path'])
    color_txt = universal['color_back']
    color_but = universal['color_btn']
    mini_frame['background'] = color_txt    
    mini_frame.protocol('WM_DELETE_WINDOW', process.nClose)
    mini_frame.resizable(0,0)
    mini_frame.title('Progress')

    # dsplay the training device 
    tk.Label(mini_frame, text='Training in '+ str(device) +'...', font='Arial', bg = color_txt).place(x = 10, y = 5)

    # initial epochs count(0/epochs)
    tk.Label(mini_frame, text='Epochs {}/{} completed...'.format(0, epochs), font = 'Arial', bg = color_txt).place(x = 10, y = 30)

    # progress bar
    progress = ttk.Progressbar(mini_frame, orient = 'horizontal', maximum = epochs, mode = 'determinate')
    progress.place(x = 10, y = 70, width=280)
    progress['value'] = 0
    progress.update()

    # Interrupt button
    button_interrupt = tk.Button(mini_frame, text='Interrupt', bg = color_but, command = mini_frame.destroy)
    button_interrupt.place(x = 100, y = 100, height=30, width=100)

    # load the network from the models folder
    model = models.vgg19(pretrained=True).features
    # freeze all VGG parameters
    for param in model.parameters():
        param.requires_grad_(False)

    # change the images to torch tensor and move to the device
    content_tensor, _, _ = process.pre_process(content_img)
    style_tensor, style_mean, style_var = process.pre_process(style_img, shape = content_tensor.shape[2:])

    content_tensor = content_tensor.to(device)
    style_tensor = style_tensor.to(device)

    # layers used in the model with weights to transfer the style
    style_layer_weights = {'conv1_1': 1.,
                           'conv2_1': 0.75,
                           'conv3_1': 0.2,
                           'conv4_1': 0.2,
                           'conv5_1': 0.2}
    content_layer_weights = {'conv4_2': 1}

    # create target tensor from content tensor, set the gradient on and move it to the device
    target_tensor = content_tensor.clone().requires_grad_(True).to(device)

    # extracting the layers from the network for the content and style tensor
    content_layers = process.layer_extract(model, content_tensor)
    style_layers = process.layer_extract(model, style_tensor)
    # gram matrix for the style layers
    style_layers = {layer: process.gram_matrix(style_layers[layer]) for layer in style_layers} 

    # set the optimizer
    optimizer = torch.optim.Adam([target_tensor], lr=0.01)

    for i in range(1, epochs+1):

        # extract the layers from the network for the target tensor
        target_layers = process.layer_extract(model, target_tensor)

        # calculate the individual loss for content and style
        content_layer_loss = process.calculate_layer_loss(target_layers, content_layers, content_layer_weights)
        style_layer_loss = process.calculate_layer_loss(target_layers, style_layers, style_layer_weights, style = True)
    
        # total loss of the system
        total_loss = alpha * content_layer_loss + beta * style_layer_loss
    
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Updating the info in progress bar window
        tk.Label(mini_frame, text='Epochs {}/{} completed...'.format(i, epochs), font='Arial', bg = color_txt).place(x = 10, y = 30)
        progress['value'] = i
        progress.update()

    # get the output image from the output(target) tensor
    target_img = process.denorm_tensor(target_tensor, style_mean, style_var)

    # convert the image type from numpy to PIL object
    target_img = Image.fromarray(np.uint8(target_img * 255))

    # Destroy the progress bar window
    mini_frame.destroy()

    return target_img
