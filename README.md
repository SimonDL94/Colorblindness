# Colorblindness

### Intro
Colorblindness includes necessary code to preprocess "red-green" **"Ishihara"** color blind tests in order to make the number / figure being diplayed become visible. 

##

### src/ImageProcessing.py
The ImageProcessing.py class contains the necessary preprocessing need in order to obtain a binary black-white contrast image diplaying the number / figure being displayed on the color blindness test

## CNN Model
Other files are included to train a CNN model trained on MNSIT data that is able to recognise only "single digits" being displayed on the "Ishihara" color blindness test (after having preprocessed the iamge using the processing steps in the ImageProcessing.py class
This means that the example CNN model being displayed will not be able to recognised "Ishihara" tests with double digits of other figures than single digits.

### src/DataLoading.py
This class includes the necessary steps to load images from the open source MNIST dataset from the Keras library; in addition the images are pixel inverted in order to increase the dataset and make a combined dataset of both "white" numbers on "black" background and vice versa "black" numbers on "white" background

### src/ModelTraining.py
This class includes the necessary steps to train an convolutional neural network using the Keras library. Good reference on intro to conv nets in Keras: https://victorzhou.com/blog/keras-cnn-tutorial/

### requirements.txt
this file contains the necessary python packages to be installed in your virtual environment in order to make use of the code

### trainCNN.py
This script can be run to train your own CNN model based on the MNIST dataset and save it under the src/ folder

### main.py
this contains the main code to run in order to preprocess an new "Ishihara" color blind test image, feed it to the trained CNN model and trying to predict the number being displayed inside the test
