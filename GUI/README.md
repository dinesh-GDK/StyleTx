# StyleTx GUI

This is the GUI version of the **StyleTx** python package.

## Installation
You can clone this repository and run the `main.py` file which is inside the folder `GUI`, in the terminal to open the GUI.

`git clone https://github.com/dinesh-GDK/StyleTx.git`

## Requirements
Python3

**StyleTx** depends on the following python packages
```
torch==1.5.0
torchvision==0.6.0
numpy==1.19.0
pillow==7.1.2
tqdm==4.46.1
```
All the requirements stated above are used in creating the project, lower versions of the requirements may or may not work.\
Also, using GPU will significantly reduce the run time of the script. Make sure to get `torch` and `torchvision` version that supports GPU.

## Implementation

Open the `main.py` file which is inside the folder `GUI`, in the terminal, upload the **content** and **style** images and generate the **output** image. You can change the parameters using the **Advanced Settings** options. You can preview and save your images.

## Example
![alt text](https://github.com/dinesh-GDK/StyleTx/blob/master/images/Result.png)

## References
The complete theory behind the **StyleTransfer** can be found in this [link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/dinesh-GDK/StyleTx/blob/master/LICENSE.txt)

## Resources
https://github.com/udacity/deep-learning-v2-pytorch/tree/master/style-transfer