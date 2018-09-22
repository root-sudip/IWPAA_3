"""
Developer: Sudip Das
Licence : Indian Statistical Institute
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

from keras.models import model_from_json

import csv

from keras.utils import np_utils
import operator as op

import matplotlib.pyplot as plt


class mlp:
	def __init__(self,no_epoch):
		print('mlp ceated...')
		self.no_epoch = no_epoch

	def load_data(self,filename,rows,cols):
		X = np.zeros([rows,cols])
		Y = np.zeros([rows])
		i = 0
		with open(filename) as file:
			csv_reader = csv.reader(file)
			for row in csv_reader:
				j = 0
				for digit in row[:cols]:
					X[i][j] = digit
					j += 1
				Y[i] = row[cols]
				i += 1
			file.close()

		return X, Y


	def create_model(self):
		self.model = Sequential()
		self.model.add(Dense(output_dim=6, input_dim=2))
		self.model.add(Activation("relu"))
		# self.model.add(Dense(output_dim=10, input_dim=10))
		# self.model.add(Activation("relu"))
		self.model.add(Dense(output_dim=2))
		self.model.add(Activation("softmax"))
		sgd = optimizers.SGD(lr=0.1, momentum=0.25)
		self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
		self.model.summary()
		
	def train_model(self, filename, rows, cols):
		
		self.best_accuracy = 0.0
		losses = []
		accuracy = []

		X_Train, Y_Train = self.load_data(filename, rows, cols)
		print(X_Train,Y_Train)
		Y_Train = np_utils.to_categorical(Y_Train)
		print(Y_Train)

		plt.ion()

		for i in range(0, self.no_epoch): #epochs
			print('Iteration == ',i)
			self.accuracy_measures = self.model.fit(X_Train, Y_Train, nb_epoch=1, batch_size=8)
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


			plt.pause(.001)
			###################################

			self.save_model()  #wait save

		print('After ',self.no_epoch,' epoch best accuracy is : ',self.best_accuracy)
		plt.pause(100)


	def save_model(self):
		model_json = self.model.to_json()
		with open("model_mlp.json", "w") as json_file:
			json_file.write(model_json)
		self.model.save_weights("model_mlp.h5")

	def load_mode(self):
		json_file = open('model_mlp.json', 'r')
		self.model = json_file.read()
		json_file.close()
		self.model = model_from_json(self.model)
		self.model.load_weights("model_mlp.h5")

	def test_model(self,filename,rows,cols):
		
		X_Test, Y_Test = self.load_data(filename,rows, cols) #total samples, total features(excluding label)

		self.classes = self.model.predict_classes(X_Test, batch_size=120)

		#get accuration

		self.accuration = np.sum(self.classes == Y_Test)/float(rows) * 100

		print ('Test Accuration : ',str(self.accuration),'%')

ob = mlp(no_epoch = 700)
ob.create_model()
ob.train_model(filename='ring_dist.csv',rows=100,cols=2)


#ob.load_mode()
#ob.test_model('xor_type_test.csv',40,2)
