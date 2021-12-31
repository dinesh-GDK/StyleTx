import styletx
import PIL
import matplotlib.pyplot as plt

content = PIL.Image.open("./images/space_needle.jpg")
style = PIL.Image.open("./images/Mona_Lisa.jpg")

content.show()
style.show()

output = styletx.StyleTransfer(content, style, epochs=2)

output.show()