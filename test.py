import styletx
import PIL
import matplotlib.pyplot as plt

content = PIL.Image.open("./images/Result.png")
style = PIL.Image.open("./images/Result.png")

content.show()
style.show()

output = styletx.StyleTransfer(content, style, epochs=2)

output.show()