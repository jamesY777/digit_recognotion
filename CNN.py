####Large Convolutional Neural Network for MNIST######

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Reshape the image to 3D width=28, height=28, channels=1. Channels is the colors of the pixel, in case of RGB, the channels = 3 
# has 3 image inputs for every color image. Here the gray scale, pixel dimension is set to 1.
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# transfer a number to one hot encoding of the class vales. 
# eg. 1 -> [0,1,0,0,0,0,0,0,0]
# eg. 2 -> [0,0,1,0,0,0,0,0,0]
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the CNN model with complex layers
    # 1. Convolutional layer Convolution2D. This layer has 30 feature maps, which with the size of 5*5.
    # 2. Pooling layer that takes the max called MaxPooling2D. This configured with a pool size of 2*2.
	# 3. Convolutional layer Convolution2D. This layer has 15 feature maps, which with the size of 3*3.
    # 4. Pooling layer that takes the max called MaxPooling2D. This configured with a pool size of 2*2.
    # 5. Regularization layer using dropout called Dropout. This is configured to randomly exclude 20% of the nerons in the layer.
    # 6. Flatten layer to converts the 2D matrix data to a vector.
    # 7. Connected layer with 128 neurons and retifier activation function.
	# 8. Connected layer with 50 neurons and retifier activation function.
    # 9. Finally, the output layer has 10 neurons for the 10 classes and a softmax activation function to output probability-like prediction for each class.

def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model3 = larger_model()

model3.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

scores = model3.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

model3.save('sample_model/CNN.h5')