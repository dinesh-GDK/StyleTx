import tkinter as tk
from tkinter import filedialog
import PIL
from  PIL import ImageTk
from functools import partial

from StyleTransfer import *

universal = {'color_back': '#87C7B9', 'color_btn': '#34A88E'}
class Frame(tk.Tk):
    '''Wrapper GUI for Style Transfer(StyleTx)'''

    def __init__(self):

        # initialize the main window
        super(Frame, self).__init__()
        self.resizable(0, 0)
        self.title('StyleTx')
        self.geometry('520x250')
        self.color_txt = universal['color_back']
        self.color_but = universal['color_btn']
        self['background'] = self.color_txt 

        # files types used
        self.ftypes = [('All files', '*'), ('JPG', ('*.jpg', '*.jpeg')),
                       ('PNG', '*.png'), ('BMP', '*.bmp'), ('TIFF', '*.tiff')]

        # variables to pass to functions using buttons
        self.variables = {1: 'self.image_content', 2: 'self.image_style', 3: 'self.image_output'}
        
        # inputs for StyleTransfer function
        self.alpha = 1
        self.beta = 10
        self.epochs = 10

        # display all buttons and text in the main window
        self.text()
        self.buttons()

    def text(self):
        '''Text used in the main window'''

        tk.Label(self, text = 'Content Image', font='Gothic 18', bg = self.color_txt).place(x = 35, y = 20)
        tk.Label(self, text = 'Style Image', font='Forte 18', bg = self.color_txt).place(x = 350, y = 20)
        tk.Label(self, text = 'Output Image', font='Arial 15', bg = self.color_txt).place(x = 190, y = 120)

    def buttons(self):
        '''Buttons used in main window'''

        self.button_upload_content = tk.Button(self, text='Upload', bg = self.color_but, command = partial(self.upload, self.variables[1]))
        self.button_upload_content.place(x = 10, y = 50, height = 30, width = 100)

        self.button_display_content = tk.Button(self, text='Preview', bg = self.color_but, command = partial(self.display, self.variables[1]))
        self.button_display_content.place(x = 120, y = 50, height = 30, width = 100)

        self.button_upload_style = tk.Button(self, text='Upload', bg = self.color_but, command = partial(self.upload, self.variables[2]))
        self.button_upload_style.place(x = 300, y = 50, height = 30, width = 100)

        self.button_display_content = tk.Button(self, text='Preview', bg = self.color_but, command = partial(self.display, self.variables[2]))
        self.button_display_content.place(x = 410, y = 50, height = 30, width = 100)

        self.button_styleTransfer = tk.Button(self, text='Generate', bg = self.color_but, command = self.styleTransfer)
        self.button_styleTransfer.place(x = 75, y = 150, height = 30, width = 100)

        self.button_display_output = tk.Button(self, text='Preview', bg = self.color_but, command = partial(self.display, self.variables[3]))
        self.button_display_output.place(x = 200, y = 150, height = 30, width = 100)

        self.button_save_output = tk.Button(self, text='Save', bg = self.color_but, command = self.save_output)
        self.button_save_output.place(x = 325, y = 150, height = 30, width = 100)

        self.button_settings = tk.Button(self, text='Adavnced Settings', bg = self.color_but, command = self.advancedSettings)
        self.button_settings.place(x = 190, y = 200, height = 30, width = 140)

    def upload(self, var):
        '''Upload the files and load it in corresponding variable
        INPUTS: var - string containing variable name (type: string)
        '''
        self.closeFrames()
        filename = tk.filedialog.askopenfile(title='Upload', filetypes=self.ftypes)
       
        try:
            exec(var + '= PIL.Image.open(filename.name)')          # load the image   
        except PIL.UnidentifiedImageError:                         # not an image, so PIL cannot load it
            self.error_msg = 'Enter an valid image'
            self.exceptionFrame()
        except AttributeError:                                     # if user does not choose a file and closes the window
            pass

    def display(self, var):
        '''Display the image from the variable
        INPUTS: var - string containing variable name (type: string)
        '''
        self.closeFrames()
        try:
            img = eval(var)                                        # load the image into var
        except AttributeError:                                     # if image is not available
            self.error_msg = 'Upload an valid image'               # if the image(content/style) is not uploaded
            if 'output' in var:
                self.error_msg = 'Generate the image'              # if output image is not generated
            self.exceptionFrame()
            return None                                            # if any of the exceptions occur do not move fwd 

        self.mini_frame = tk.Toplevel(self)
        self.mini_frame.resizable(0, 0)
        
        # set title of the display frame
        if 'content' in var:
            self.mini_frame.title('Content Image')
        elif 'style' in var:
            self.mini_frame.title('Style Image')
        else:
            self.mini_frame.title('Output Image')

        # resize the image if the width of the image is too large
        if img.size[0] > 1000:
            wsize = 1000
            wpercent = (wsize / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((wsize, hsize), Image.ANTIALIAS)

        # resize the image if the height of the image is too large
        if img.size[1] > 700:
            hsize = 700
            hpercent = (hsize / float(img.size[1]))
            wsize = int((float(img.size[0]) * float(hpercent)))
            img = img.resize((wsize, hsize), Image.ANTIALIAS)

        img = PIL.ImageTk.PhotoImage(img)
        panel = tk.Label(self.mini_frame, image=img)
        panel.image = img
        panel.pack()

    def styleTransfer(self):
        '''Implement the StyleTransfer function'''

        self.closeFrames()
        self.error_msg = ''

        # check if content image is available
        try:
            self.image_content
        except AttributeError:
            self.error_msg = 'Upload a valid content image'

        # check if style image is available
        try:
            self.image_style
        except AttributeError:
            self.error_msg += '\nUpload a valid style image'
            self.exceptionFrame()
            return None

        try:
            self.image_output = StyleTransfer(self.image_content, self.image_style, universal, self.alpha, self.beta, self.epochs)
        except:
            None                                                # if process breaks due to interrupt button

    def save_output(self):
        '''Save the output image'''

        self.closeFrames()

        # check whether the output image is generated
        try:
            self.image_output
        except AttributeError:
            self.error_msg = 'Generate the image'
            self.exceptionFrame()
            return None
        
        filename = tk.filedialog.asksaveasfile(mode='w', defaultextension='.png', filetypes=self.ftypes)

        # if user closes the save as window without saving the image
        try:
            self.image_output.save(filename.name)
        except AttributeError:
            pass

    def advancedSettings(self):
        '''Implement the advanced settings features''' 
        
        self.closeFrames()
        self.settings_frame = tk.Toplevel(self)
        self.settings_frame.geometry('270x200')
        self.settings_frame['background'] = self.color_txt
        self.settings_frame.resizable(0, 0)
        self.settings_frame.title('Advanced Settings')

        # text in Advanced Settings window
        tk.Label(self.settings_frame, text='alpha', font='15', bg = self.color_txt).place(x = 50, y = 10)
        tk.Label(self.settings_frame, text='beta', font='15', bg = self.color_txt).place(x = 50, y = 50)
        tk.Label(self.settings_frame, text='epochs', font='15', bg = self.color_txt).place(x = 50, y = 90)
        tk.Label(self.settings_frame, text='All values must be integers', bg = self.color_txt).place(x = 70, y = 120)

        # create dummy variables for StyleTransfer variables
        dalpha = tk.StringVar()
        dalpha.set(str(self.alpha))
        dbeta = tk.StringVar()
        dbeta.set(str(self.beta))
        depochs = tk.StringVar()
        depochs.set(str(self.epochs))

        # entry boxes that loads value into the dummy variables
        tk.Entry(self.settings_frame, textvariable = dalpha,  width = 10).place(x = 150, y = 10)
        tk.Entry(self.settings_frame, textvariable = dbeta, width = 10).place(x = 150, y = 50)
        tk.Entry(self.settings_frame, textvariable = depochs, width = 10).place(x = 150, y = 90)

        # Apply Changes button
        self.button_apply = tk.Button(self.settings_frame, text='Apply Changes', bg = self.color_but,
                                      command = partial(self.applyChanges, dalpha, dbeta, depochs))
        self.button_apply.place(x = 75, y = 150, height = 30, width = 120)

    def applyChanges(self, dalpha, dbeta, depochs):
        '''Changes the values of the StyleTransfer variables if the values are of type int
        INPUTS: dalpha - dummytext box variable of self.alpha (type: string)
                dbeta - dummytext box variable of self.beta (type: string)
                depochs - dummytext box variable of self.epochs (type: string)
        '''
        
        self.error_msg = 'Enter valid inputs'
        for i in [dalpha, dbeta, depochs]:
            if ('.' in i.get()):                            # check for type float
                self.exceptionFrame()
                return None
            try:    
                int(i.get())                                # check for other types except int
            except ValueError:
                self.exceptionFrame()
                return None
        
        # change the StyleTransfer variables values to dummy variables values
        self.alpha = int(dalpha.get())              
        self.beta = int(dbeta.get())
        self.epochs = int(depochs.get())
        self.settings_frame.destroy()
        
    def exceptionFrame(self):
        '''To display error messages'''

        self.closeFrames()
        self.mini_frame = tk.Toplevel(self)        
        self.mini_frame.geometry('250x50')
        self.mini_frame['background'] = self.color_txt
        self.mini_frame.resizable(0,0)
        self.mini_frame.title('ERROR')
        error = tk.Label(self.mini_frame, text = self.error_msg, font = '15', bg = self.color_txt)
        error.place(x=10, y=0)

    def closeFrames(self):
        '''Closes the mini_frames if available'''

        try: self.mini_frame.destroy()     
        except AttributeError: pass

if __name__ == '__main__':
    main = Frame()
    main.mainloop()
