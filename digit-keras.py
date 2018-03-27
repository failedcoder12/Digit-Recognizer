import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#Loading test and train data
(mnist_train_images, mnist_train_labels) ,(mnist_test_images, mnist_test_labels) = mnist.load_data()

#Loading test and train data as float32 values
train_images = mnist_train_images.reshape(60000, 784)
test_images = mnist_test_images.reshape(10000, 784)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# Normalizing it to 0-1 from 0-255
train_images /= 255
test_images /= 255

# Converting them to one-hot format
train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)

import matplotlib.pyplot as plt

def display_sample(num):
	#print in one hot format
	print(train_labels[num])

	#converting label to a number
	label = train_labels[num].argmax(axis=0)

	#reshape image in 28*28 image
	image = train_images[num].reshape([28,28]) 

	#plot number
	plt.title('Sample: %d Label: %d' %(num,label))
	plt.imshow(image, cmap=plt.get_cmap('gray_r'))
	plt.show()

display_sample(1234)

# Adding layers

model = Sequential()

#input layer of 784 size feds into a ReLU with 512 node
model.add(Dense(512, activation='relu', input_shape=(784,)))

#Adding Dropout
model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

#Adding Dropout to prevent Overfitting
model.add(Dropout(0.2))

#Then goes to a 10 nodes with softmax applied
model.add(Dense(10,activation='softmax'))

#Print about neural network
print(model.summary())

#setting up loss function and optimizer and running it
model.compile(loss='categorical_crossentropy',
	optimizer=RMSprop(),
	metrics=['accuracy'])

#How data is trained
history = model.fit(train_images, train_labels,
	batch_size=100,
	epochs=10,
	verbose=2,
	validation_data=(test_images,test_labels))

print(history)

# getting accuracy
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])
