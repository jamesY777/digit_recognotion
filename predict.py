import keras
from numpy import expand_dims
from numpy import asarray
from numpy import moveaxis
from PIL import Image

# Load the trained model
model = keras.models.load_model('sample_model/CNN_1.h5')

img_number = Image.open('sample/4.png')

#convert the image to grayscale
img_number = img_number.convert(mode='L')
img_number = img_number.resize((28, 28))
img_number.show()
# convert to numpy array
data = asarray(img_number)

# add channels last
data_last = expand_dims(data, axis=2) #axis=0, to add dim third (last)
data_last = data.reshape((1,28,28,1)) #Reshape the data to be single entry

result = model.predict(data_last)
print(f'The image is a {result.argmax()}')