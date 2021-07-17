from tkinter import *
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import ImageTk, Image
import io
from numpy import asarray
import keras
from numpy import expand_dims

# Basic window
window = Tk()
window.geometry('300x300+400+300')
window.title('Hand Written Digit Prediction')

# Canvas for drawing
canvas = Canvas(window)
canvas.pack()

# Drawing application
current_x, current_y = 0, 0
def locate_xy(event):
    global current_x, current_y
    current_x, current_y = event.x, event.y

def draw(event):
    global current_x, current_y
    canvas.create_oval((current_x, current_y, event.x, event.y), width=15, fill="white")
    current_x, current_y = event.x, event.y

canvas.bind('<Button-1>', locate_xy)
canvas.bind('<B1-Motion>', draw)

# Seperate
separator = ttk.Separator(window, orient='horizontal')
separator.pack(fill='x')

# Button for predict
# Load trained model
model = keras.models.load_model('sample_model/CNN_1.h5')
def predict():
    ps = canvas.postscript(colormode='color')
    img_number = Image.open(io.BytesIO(ps.encode('utf-8')))
    img_number = img_number.convert(mode='L')
    img_number = img_number.resize((28, 28))
    # img_number.save('./test.jpg')
    data = 1-asarray(img_number)/255
    data_last = expand_dims(data, axis=2) #axis=0, to add dim third (last)
    data_last = data.reshape((1,28,28,1)) #Reshape the data to be single entry
    result = model.predict(data_last)
    showinfo(
        title='Predict Result',
        message = f'This is a {result.argmax()}'
    )
btn_predict = Button(window, text='Predict', command=predict).pack(side='left',fill='x', expand=True)

# Button for reset
btn_reset = Button(window, text='Reset', command=lambda: canvas.delete("all")).pack(side='right',fill='x',expand=True)

window.mainloop()


