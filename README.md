# hand_write_digit_recognotion
 
## Summary
Here I established a very simple CNN (Convolutional Neural Network) model to recognize handwritten digit 0~9.

This repo contains following:
- CNN.py set up the model and traning the data usig MNIST data set which contains 60000 handwritten digits.
- sample_model collects the trianed model from generated using CNN.py file. The model is saved as .h5 files.
- sample_digit collects the test image file (png) of handwritten digit for single test.
- predict.py reads the .h5 trained model and predicts the digit of test image file.
- draw_application.py is a small UI allows user to write the digit and using my trained model to predict the result. 

## Basic Neural Network
NN is a deep learning model. It takes input data and conducted a series of calculations to produce the output, then compare the output with the true result data. The difference between output and the true result is call Loss. The training process is to adjust the cacluations parameters to minimize the Loss. Those parameters will be saved as the trained model, so they can be reused in the future to do predict.

Basically, each neuron would represent a number, input data is a series of neuron can represent a vector (list of numbers). NN layer is matrix with numbers, those nubmers are called parameters. Each layer will conduct a matrix calculation with some Activation Functions (make model non-linear). Following image shows how a basic NN works: (image source https://gfycat.com/gifs/search/neural+network)


![](/repo_image/NN_demo.gif)


## CNN

CNN (Convolutional Neural Network) is a widely used deep neural network, typically for analysis of image data. 

Image data can be represent as a 3D matrix (2D pixels + 1D color channel). e.g I can describe a image with 3*3 pixels of a vertial line in middle using matrix: [[0,1,0],[0,1,0],[0,1,0]]. 1 = white, 0 = black.

CNN uses such filter (small image matrix) to scan the image data which can extract the features (lines, circles, etc).
Below is an example of filter extract the edge of a digit image (image source from https://deeplizard.com/learn/video/YRhxdVk_sIs)

![](/repo_image/filter_demo2.png)
![](/repo_image/filter_demo1.png)

Then, Max Pool layer will be used to find the most siginficant features in the image. (image source from https://nico-curti.github.io/NumPyNet/NumPyNet/layers/maxpool_layer.html)
![](/repo_image/maxpool_demo.gif)

## My model

This is a very simple CNN model with 9 layers, follow is the description of each layer.
I have trained it with 10 epochs (times).
```
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
```

## draw application 
To simply demo the performance of the training model, I build a draw applicaton. You may use it to draw your own number and predict it.
![](/repo_image/draw_application.gif)


  *Note: the UI is built using tkinter canvas and the canvas file requires installation of **ghostscript** so it can be convert to a image. Suggest add the gs bin folder in the local path. Download gs at: https://www.ghostscript.com/*