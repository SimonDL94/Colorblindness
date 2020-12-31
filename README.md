# Colorblindness

### Intro
Colorblindness includes necessary code to preprocess "red-green" **"Ishihara"** color blind tests in order to make the number / figure being diplayed become visible. 

![alt text](https://github.com/SimonDL94/Colorblindness/blob/master/images/processingImage.png | width=100)

### src/ImageProcessing.py
The ImageProcessing.py class contains the necessary preprocessing need in order to obtain a binary black-white contrast image diplaying the number / figure being displayed on the color blindness test. The ImageProcessing.py contains the necessary elements to obtain a binary black-white pixel image:

![alt text](https://github.com/SimonDL94/Colorblindness/blob/master/images/processingImageStep1.png)

## Convolutional Neural Network (CNN) model
A CNN is a machine learning model that can be specifically used to perform an image classification problem such as classifying the digits on the "Ishihara" color blindness test; more info: https://victorzhou.com/blog/keras-cnn-tutorial/. The CNN being used here also includes the necessary Pooling, Batch Normalization layers to improve training & performance.
:::: more info on Pooling: https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/; more on Batch Normalization: https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/;

A CNN model can be trained on the famous MNIST dataset of handwritten "single" digits tin order to recongise the "single" digits being displayed (after having preprocessed the color blindimage with ImageProcessing.py). This means that the example CNN model being used here will not be able to recognised "Ishihara" tests with double digits of other figures than single digits.

### src/DataLoading.py
This class includes the necessary steps to load images from the open source MNIST dataset from the Keras library; in addition the images are pixel inverted in order to increase the dataset and make a combined dataset of both "white" numbers on "black" background and vice versa "black" numbers on "white" background

### src/ModelTraining.py
This class includes the necessary steps to train an convolutional neural network using the Keras library. Good reference on intro to conv nets in Keras: 

### requirements.txt
this file contains the necessary python packages to be installed in your virtual environment in order to make use of the code

### trainCNN.py
This script can be run to train your own CNN model based on the MNIST dataset and save it under the src/ folder

### main.py
this contains the main code to run in order to preprocess an new "Ishihara" color blind test image, feed it to the trained CNN model and trying to predict the number being displayed inside the test
