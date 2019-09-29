# Face-Recognition

This is a very basic Face Recognition system that utilizes deep learning to classify between different faces. The project utilizes Convolutional Neural Networks for Identifying different person. The CNN Architecture used in this project is loosely based on Oxford's Visual Geometery Group's Deep Face Architechture shown :point_down:


![picture alt](https://i0.wp.com/sefiks.com/wp-content/uploads/2018/08/vgg-face-model.png?ssl=1 "VGG 16")

![picture alt](https://i0.wp.com/sefiks.com/wp-content/uploads/2019/04/vgg-face-architecture.jpg?ssl=1 "VGG 16")


# ALGORITHM:

1. Load raw images

2. Convert raw images to categorical data

3. Create a Neural Network

4. Train the Network

5. Save the Network. Load the Network

6. Load video from Webcam

7. Convert frames to Gray scale

8. Detect Faces inside the Frame

9. Predict, using the trained Network

10. Optimize the Network according to necessity


# HOW TO USE(tested on Ubuntu 16.04):
1. Git clone this repository

2. run setup.sh
	1. First, it will create a virtual environment in the directrory you are working on.
	
	2. Then it will also install all the necessary modules required for this project.
	
	3. At last, it will activate the Virutlal environment.

3. Create a folder in the image folder with the name of the person you wish to Identify

4. Add around 30-50 images of that person's face

5. Run the Traning on all the newly added images with train.py file

6. Test the model with test.py file


# Tutorials:

1. ANN Tutorial Blog - 

2. ANN sample codes - https://github.com/DixitIshan/Deep_Learning_with_Keras/tree/master/ANN's

3. CNN Tutorial Blog -

4. CNN sample codes - https://github.com/DixitIshan/Deep_Learning_with_Keras/tree/master/CNN's


# References:

1. https://viblo.asia/p/facial-recognition-system-face-recognition-Ljy5Vr6j5ra#_prepare-trainingtesting-data-0


# Notes:

* This Project is completed on a device with 6th gen Intel i5 Processor, 8 Gigs DDR3 RAM, and 2 Gigs Nvidia 940 Mx Graphics Card. Tensorflow GPU version is used for Training the Network.
