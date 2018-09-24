import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten

import csv
from sklearn.datasets import load_iris
from keras.utils import np_utils
import operator as op

from keras.datasets import mnist

from keras import optimizers


import matplotlib.pyplot as plt
from keras.models import model_from_json

class mlp:
	def __init__(self,no_epoch):
		print('mlp ceated...')
		self.no_epoch = no_epoch

	def load_data(self):

		(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
		X_train = X_train.reshape(60000, 28, 28, 1)
		X_test = X_test.reshape(10000, 28, 28, 1)
		
		print('X_train shape : ', X_train.shape)
		print('X_test shape : ', X_test.shape)

		Y_train = np_utils.to_categorical(Y_train)
		#Y_test = np_utils.to_categorical(Y_test)

		return X_train, Y_train, X_test, Y_test

		
	def create_model(self):
		
		self.model = Sequential()
		self.model.add(Conv2D(256, (3, 3), padding='same',input_shape=(28,28, 1)))
		self.model.add(Activation("relu"))
		self.model.add(Conv2D(512, (3, 3), padding='same'))
		self.model.add(Activation("relu"))
		#self.model.add(Conv2D(256, (3, 3), padding='same'))
		#self.model.add(Activation("relu"))
		self.model.add(Conv2D(512, (3, 3), padding='same'))
		self.model.add(Activation("relu"))
		self.model.add(Flatten())
		self.model.add(Dense(output_dim=40))
		self.model.add(Activation("relu"))
		self.model.add(Dense(output_dim=10))
		self.model.add(Activation("softmax"))
		sgd = optimizers.SGD(lr=0.001, momentum=0.0005)
		self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
		self.model.summary()
		
	def train_model(self):
		self.best_accuracy = 0.0
		plt.ion()

		losses = []
		accuracy = []

		X_train, Y_train, _, _ = self.load_data()

		for i in range(0, self.no_epoch): #epochs
			print('Iteration == ',i)
			self.accuracy_measures = self.model.fit(X_train, Y_train, nb_epoch=1, batch_size=128)
			#print(self.accuracy_measures.history.keys())

			self.iter_accuracy = op.itemgetter(0)(self.accuracy_measures.history['acc'])
			self.iter_loss = op.itemgetter(0)(self.accuracy_measures.history['loss'])
			losses.append(self.iter_loss)
			accuracy.append(self.iter_accuracy)

			if (self.best_accuracy < self.iter_accuracy):
				self.best_accuracy = self.iter_accuracy

			#ploting the accuracy and error	
			plt.subplot(2, 1, 1)
			plt.plot(losses, "r")
			plt.xlabel('Epochs ')
			plt.ylabel('Loss')

			plt.subplot(2, 1, 2)
			plt.plot(accuracy, "b")
			plt.xlabel('Epochs ')
			plt.ylabel('Accuracy')

			self.save_model()  #weight save

			plt.pause(.001)
			###################################
			

		print('After ',self.no_epoch,' epoch best accuracy is : ',self.best_accuracy)
		plt.pause(100)



	def save_model(self):
		model_json = self.model.to_json()
		with open("model_MNIST_MLP.json", "w") as json_file:
			json_file.write(model_json)
		self.model.save_weights("model_MNIST_MLP.h5")

	def load_model(self):
		json_file = open('model_MNIST_MLP.json', 'r')
		self.model = json_file.read()
		json_file.close()
		self.model = model_from_json(self.model)
		self.model.load_weights("model_MNIST_MLP.h5")



	def test_model(self):


		_,_, X_test, Y_test = self.load_data()
		#print(X_test.shape, Y_test)
		classes = self.model.predict_classes(X_test, batch_size=10000)

		self.test_dim = Y_test.shape
		print('Test dimention : ',self.test_dim)
		accuracy = np.sum(classes == Y_test)/self.test_dim * 100
		print(np.sum(classes == Y_test))
		print ('Test Accuracy : ',str(accuracy),'%')
		

tt = input('train/test')

ob = mlp(no_epoch = 1200)
ob.create_model()
if tt == 'train':
	ob.train_model()

elif tt == 'test':
	ob.load_model()
	ob.test_model()
