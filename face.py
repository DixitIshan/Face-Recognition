import cv2
import os
import numpy as np

from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

filepath = 'names.txt'
people = []
with open(filepath) as fp:
   line = fp.readline()
   while line:
       people.append(line.strip())
       line = fp.readline()

num_classes = len(people)

img_data_list = []
labels = []
valid_images = [".jpg",".gif",".png"]

for index, person in enumerate(people):
	print(index)
	dir_path = 'image/' + person
	for img_path in os.listdir(dir_path):
		name, ext = os.path.splitext(img_path)
		if ext.lower() not in valid_images:
			continue

		img_data = cv2.imread(dir_path + '/' + img_path)
		img_data=cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
		img_data_list.append(img_data)
		labels.append(index)

	img_data = np.array(img_data_list)
	img_data = img_data.astype('float32')

	labels = np.array(labels ,dtype='int64')
	img_data /= 255.0
	img_data= np.expand_dims(img_data, axis=4)

Y = np_utils.to_categorical(labels, num_classes)

x,y = shuffle(img_data,Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)