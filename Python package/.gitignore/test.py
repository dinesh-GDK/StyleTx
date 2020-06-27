from styletx import StyleTransfer
import PIL
import matplotlib.pyplot as plt

content = PIL.Image.open("E:\Projects\images\space_needle.jpg")
style = PIL.Image.open("E:\Projects\images\Mona_Lisa.jpg")

content.show()
style.show()

output = StyleTransfer(content, style, epochs=2)

output.show()