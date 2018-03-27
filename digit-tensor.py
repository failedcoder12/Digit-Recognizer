import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#starting session

sess = tf.InteractiveSession()

#dataset in mnist
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import matplotlib.pyplot as plt

#function to display image
def display_sample(num):
	#print in one hot format
	print(mnist.train.labels[num])

	#converting label to a number
	label = mnist.train.labels[num].argmax(axis=0)

	#reshape image in 28*28 image
	image = mnist.train.images[num].reshape([28,28]) 

	#plot number
	plt.title('Sample: %d Label: %d' %(num,label))
	plt.imshow(image, cmap=plt.get_cmap('gray_r'))
	plt.show()

display_sample(1234)

import numpy as np

#convert images to one dimen array
images = mnist.train.images[0].reshape([1,784])

#concatenate it to images all the images to a single array
for i in range(1,500):
	images = np.concatenate((images, mnist.train.images[i].reshape([1,784])))

plt.imshow(images, cmap=plt.get_cmap('gray_r'))
plt.show()

#Array consists of an array input 784
input_images = tf.placeholder(tf.float32, shape=[None, 784])

#output in one hot format
target_labels = tf.placeholder(tf.float32, shape=[None, 10])

#hidden layer of size 512
hidden_nodes = 512

#input given to weights and then given to hidden nodes 
input_weights = tf.Variable(tf.truncated_normal([784, hidden_nodes]))

#input biases
input_biases = tf.Variable(tf.zeros([hidden_nodes]))

#hidden nodes to 10 output nodes
hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes,10]))

#bias to 10 output nodes
hidden_biases = tf.Variable(tf.zeros([10]))

#multiplying input matrix to input weights
input_layer = tf.matmul(input_images, input_weights)

#relu activation on summation of input_layer and input_biases
hidden_layer = tf.nn.relu(input_layer + input_biases)

#matric multiplication of hidden_layer and hidden_weights
digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases

#loss function in our progress to gradient descent : 
#cross entropy which applies a logarithm scale to penalize incorrect classification
#checking output(digit_weights) against dataset labels (target_labels)
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=digit_weights, labels=target_labels))

#A gradient descent optimizer with aggressive rate 0.5 and our loss function
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)

correct_prediction = tf.equal(tf.argmax(digit_weights,1), tf.argmax(target_labels,1))

# "accuracy" then takes the average of all the classifications to produce an overall score for our model's accuracy.
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.global_variables_initializer().run()

for x in range(2000):

	batch = mnist.train.next_batch(100)
	optimizer.run(feed_dict={input_images: batch[0],target_labels: batch[1]})
	if((x+1) % 100 == 0):
		print("Trainig epoch " + str(x+1))
		print("Accuracy "+str(accuracy.eval(feed_dict={input_images: mnist.test.images, target_labels: mnist.test.labels})))


for x in range(100):

	#loading a test image and label
	x_train = mnist.test.images[x,:].reshape(1,784)
	y_train = mnist.test.labels[x,:]

	#convert to one hot lable
	label = y_train.argmax()

	#get classification from neural network
	prediction = sess.run(digit_weights, feed_dict={input_images: x_train}).argmax()

	if(prediction!=label) :
		plt.title('Prediction: %d Label: %d' % (prediction,label))
		plt.imshow(x_train.reshape(28,28),cmap=plt.get_cmap('gray_r'))
		plt.show()



