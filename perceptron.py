from __future__ import print_function
import matplotlib,sys
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import csv

from plot import *
from sklearn import preprocessing

#predicting the class respect to each sample
def predict(inputs,weights):
	activation=0.0
	for i,w in zip(inputs,weights):
		activation += i*w
	return 1.0 if activation>=0.0 else 0.0

##calculating the accuracy 
def accuracy(matrix,weights):
	num_correct = 0.0
	preds       = []
	c = 0
	w = 0
	for i in range(len(matrix)):
		pred   = predict(matrix[i][:-1],weights) # get predicted classification
		
		preds.append(pred)
		if pred==matrix[i][-1]: 
			num_correct+=1.0 
			c = c+1
		else:
			w = w + 1

	print('Number of corrects : ',c)
	print('Number of wrongs : ',w)
	return num_correct/float(len(matrix))



# each matrix row: up to last row = inputs, last row = y (classification)
def train_weights(matrix,weights,nb_epoch=10,l_rate=1.00,stop_early=True,verbose=False,do_plot=False):
	for epoch in range(nb_epoch):  #epoch

		for i in range(len(matrix)):
			prediction = predict(matrix[i][:-1],weights) # get predicted classificaion
			error      = matrix[i][-1]-prediction		 # get error from real classification
			if verbose: sys.stdout.write("Training on data at index %d...\n"%(i))
			for j in range(len(weights)): 				 # calculate new weight for each node
				if verbose: sys.stdout.write("\tWeight[%d]: %0.5f --> "%(j,weights[j]))
				weights[j] = weights[j]+(l_rate*error*matrix[i][j]) 
				if verbose: sys.stdout.write("%0.5f\n"%(weights[j]))

		cur_acc = accuracy(matrix,weights)
		print("\nEpoch %d \nWeights: "%epoch,weights)
		print("Accuracy: ",cur_acc)

		if do_plot: 
			plot(matrix=matrix,weights=weights,title="Epoch %d"%epoch)

		if cur_acc==1.0 and stop_early: break

	return weights 

#prediction of the classes 
def prediction(data,weights):
	cur_acc = accuracy(data,weights)
	print('Accuracy : ' ,cur_acc)

#feeding the data
def feed_matrix(data, weights,l_rate, nb_epoch, stop_early,do_plot=False):

	weights = train_weights(data,weights=weights,nb_epoch=nb_epoch,l_rate=l_rate,stop_early=stop_early,do_plot=plot) #updated weights
	return weights

def main():
	stop_early 		= True
	feature_vec_dim = int(input('Enter feature vector dimension : '))

	weights = list(tf.Session().run(tf.truncated_normal([feature_vec_dim+1], stddev=0.1))) #initializing the weights using truncated normal distribution
	print('Initial weights : ',weights)
	input_data = 'xor_type.csv'#input('Enter file name for training samples : ')

	file = open(input_data,"r")
	csv_file = csv.reader(file)

	features = []
	data = []

	for row in csv_file:
		data.append(1.0)
		for element in row[:]:
			data.append(float(element))
		features.append(data)
		data = []

	file_content = input('Do you want to see the file content (y/n)')
	if file_content == 'y':
		print(np.asarray(features))

	#features_shape = np.asarray(features)
	print('Data dimension : ', np.asarray(features).shape)


	nb_epoch = int(input('Enter number of epochs : '))
	l_rate = float(input('Learning rate : '))


	weights = feed_matrix(data=features, weights=weights, l_rate=l_rate, nb_epoch=nb_epoch, stop_early=stop_early, do_plot=True)
	print("Final weights : ", weights)

#xor data, irish, encoder_data, normal

if __name__ == '__main__':
	main()