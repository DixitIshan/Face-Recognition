# Face-Recognition

- - - -
  
# Architechture

This is a very basic Face Recognition system that utilizes deep learning to classify between different faces. The project utilizes Convolutional Neural Networks for Identifying different person. The CNN Architecture used in this project is loosely based on Oxford's Visual Geometery Group's Deep Face Architechture shown :point_down:


![picture alt](https://i0.wp.com/sefiks.com/wp-content/uploads/2018/08/vgg-face-model.png?ssl=1 "VGG 16")

![picture alt](https://i0.wp.com/sefiks.com/wp-content/uploads/2019/04/vgg-face-architecture.jpg?ssl=1 "VGG 16")

- - - -

# Algorithm:

![picture alt](https://github.com/DixitIshan/Face-Recognition/blob/master/screenshots/Untitled%20Diagram%20(1).jpg "Flow")

- - - -

# How to use(tested on Ubuntu 16.04):
1. Git clone this repository

2. run setup.sh
	1. First, it will create a virtual environment in the directrory you are working on.
	
	2. Then it will also install all the necessary modules required for this project.
	
	3. It will activate the Virutlal environment.
	
	4. Lastly it will create an 'image' folder in which all the training images will exist

3. Create a folder in the image folder with the name of the person you wish to Identify

4. Add around 30-50 images of that person's face

5. Run the Traning on all the newly added images with train.py file

6. Test the model with test.py file

- - - -

# Tutorials:

1. ANN Tutorial Blog - 

2. ANN sample codes - https://github.com/DixitIshan/Deep_Learning_with_Keras/tree/master/ANN's

3. CNN Tutorial Blog -

4. CNN sample codes - https://github.com/DixitIshan/Deep_Learning_with_Keras/tree/master/CNN's

- - - -

# Notes:

* This Project is completed on a device with 6th gen Intel i5 Processor, 8 Gigs DDR3 RAM, and 2 Gigs Nvidia 940 Mx Graphics Card. Tensorflow GPU version is used for Training the Network.

* If you find any bug in the code or you would like to contribute to the code in any way, please do not hesitate.
