# aicon
**aicon** is a python package that applies effects to an image using machine learning.

## Installation
You can install the aicon package using the commands given below

`pip install aicon`

**aicon** depends on the following python packages
```
PyTorch
numpy
matplotlib
PIL
tqdm
```
Also, using GPU will significantly reduce the run time of the script. Make sure to get PyTorch version that supports GPU.

## StyleTransfer
**StyleTransfer** is a function in **aicon** that takes two images namely style image and content image, and applies the effects in style image and applies it to the content image.

### Implementation

```
# import necessary packages
from aicon import StyleTransfer
from PIL import Image
import matlibplot.pyplot as plt

# import the images
content_image = Image.open('path/filename')
style_image = Image.open('path/filename')

# implement StyleTransfer
output_image = StyleTransfer(content_image, style_image, alpha=1, beta=10, epochs=500)

# plot the reults
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(content_image)
ax2.imshow(output_image)
plt.show()
```
The above code will apply the effects of the style image to content image.

### Inputs
content_image - a PIL object\
style_image - a PIL object\
alpha - a positive integer\
beta - a positive integer\
epochs - a positive integer

By default alpha = 1, beta = 10 and epochs = 500.
You can play around these values to get desired output image.

### Example
![alt text](https://github.com/dinesh-GDK/aicon/blob/master/aicon/images/Result.png)

### References
The complete theory behind the **StyleTransfer** can be found in this [link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/dinesh-GDK/aicon/blob/master/aicon/LICENSE.txt)
