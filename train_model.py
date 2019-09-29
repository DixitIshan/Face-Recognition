import cv2
import os
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import (Conv2D,
	MaxPooling2D,
	ZeroPadding2D,
	Flatten,
	Dense,
	Dropout
	)
from keras.constraints import maxnorm
from keras.optimizers import SGD

from keras.models import model_from_json, load_model

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
	dir_path = 'images/' + person
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

input_shape=img_data[0].shape
print(input_shape)

model.add(ZeroPadding2D((1,1),input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_constraint=maxnorm(3)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_constraint=maxnorm(3)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_constraint=maxnorm(3)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_constraint=maxnorm(3)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu', kernel_constraint=maxnorm(3)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu', kernel_constraint=maxnorm(3)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_constraint=maxnorm(3)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_constraint=maxnorm(3)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_constraint=maxnorm(3)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_constraint=maxnorm(3)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(2096, (7, 7), activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))

model.add(Conv2D(2096, (1, 1), activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))

epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")