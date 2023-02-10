from tkinter import *
from ImageConverter import *
import numpy as np
import ctypes

def softmax(x):
    return np.exp(x) / np.exp(x).sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class DrawableCanvas(object):

    pen_size = 10.0
    default_color = 'black'

    def __init__(self):

        self.root = Tk()
        self.root.configure(bg='grey')
        self.root.title("Handwritten number recognition")
        self.root.resizable(False, False)
        self.result = ''

        self.predict_button = Button(self.root, text='Predict', command=self.predict_drawing, height = 2, width = 15, bd = 0, bg = 'green', fg = '#FAFAFA')
        self.predict_button.grid(row=0, column=3)

        self.clear_button = Button(self.root, text='Clear', command=self.clear, height = 2, width = 15, bd = 0, bg = 'red', fg = '#FAFAFA')
        self.clear_button.grid(row=0, column=1)

        self.predict_result = Label(self.root, text=self.result, bg='grey', fg='black', font=(40))
        self.predict_result.grid(row=1, columnspan=5, pady=20)

        self.c = Canvas(self.root, bg='white', width=200, height=200)
        self.c.grid(row=2, columnspan=5)

        self.predict_title = Label(self.root, text='Prediction Result', bg='grey', fg='black', font=(30))
        self.predict_title.grid(row=0, column=5)

        self.predict_chance = Label(self.root, bg='#FAFAFA', width=30, height=15, font=(30))
        self.predict_chance.grid(row=1, rowspan=3, column=5)

        self.hidden_weights = np.load('data/weights/hiddenWeight.npy')
        self.hidden_bias = np.load('data/weights/hiddenBias.npy')
        self.hidden_weights2 = np.load('data/weights/hiddenWeight2.npy')
        self.hidden_bias2 = np.load('data/weights/hiddenBias2.npy')
        self.output_weights = np.load('data/weights/outputWeight.npy')
        self.output_bias = np.load('data/weights/outputBias.npy')

        self.setup()

        self.root.mainloop()

    def predict(self, inputs):
        hidden_layer_in = np.dot(self.hidden_weights, inputs) + self.hidden_bias
        hidden_layer_out = sigmoid(hidden_layer_in)
        hidden_layer_in_2 = np.dot(self.hidden_weights2, hidden_layer_out) + self.hidden_bias2
        hidden_layer_out_2 = sigmoid(hidden_layer_in_2)
        output_layer_in = np.dot(self.output_weights, hidden_layer_out_2) + self.output_bias
        return softmax(output_layer_in)

    def save(self):
        self.x0 = self.c.winfo_rootx() * self.scale_factor
        self.y0 = self.c.winfo_rooty() * self.scale_factor
        self.x1 = self.x0 + self.c.winfo_width() * self.scale_factor
        self.y1 = self.y0 + self.c.winfo_height() * self.scale_factor

        img = ImageGrab.grab((self.x0, self.y0, self.x1, self.y1))
        img.save('images/out.png', 'png')


    def predict_drawing(self):
        self.save()
        filename = './images/out.png'
        input = convert(filename).reshape(-1, 1)
        input = self.normalize(input)

        prediction = self.predict(input)
        self.result = np.where(prediction == np.amax(prediction))[0][0]
        self.predict_result.configure(text='The result is ' + str(self.result))
        predict_text = ''
        i = 0
        for p in prediction:
            predict_text += str(i) + ') ' + str(p) + '\n'
            i += 1

        self.predict_chance.configure(text=predict_text)

    def setup(self):
        self.original_x = None
        self.original_y = None
        self.line_width = self.pen_size
        self.color = self.default_color
        self.is_erasing = False
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.scale_factor = ctypes.windll.shcore.GetScaleFactorForDevice(0)/100

    def resize_set_image(self, image):
        size = (150, 150)
        resize = image.resize(size, Image.ANTIALIAS)
        self.nn_image = ImageTk.PhotoImage(resize)

    def clear(self):
        # self.prediction_label.delete(1.0, END)
        self.c.delete('all')
        self.predict_chance.configure(text='')

    def paint(self, event):
        self.line_width = self.pen_size
        paint_color = 'black'
        if self.original_x and self.original_y:
            self.c.create_line(self.original_x, self.original_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=True, splinesteps=37)
        self.original_x = event.x
        self.original_y = event.y

    def reset(self, event):
        self.original_x, self.original_y = None, None

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == '__main__':
    ge = DrawableCanvas()