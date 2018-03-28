import pandas as pd

feature_names =  ['party','handicapped-infants', 'water-project-cost-sharing', 
                    'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                    'el-salvador-aid', 'religious-groups-in-schools',
                    'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                    'mx-missle', 'immigration', 'synfuels-corporation-cutback',
                    'education-spending', 'superfund-right-to-sue', 'crime',
                    'duty-free-exports', 'export-administration-act-south-africa']

# Reading a text file with head as feature name
voting_data = pd.read_csv('house-votes-84.data.txt', na_values=['?'],
				names = feature_names)

#Printing initial data
print(voting_data.head())

#printing number of occurence of stuffs
print(voting_data.describe())

#drop that party which have any null value
voting_data.dropna(inplace=True)

print(voting_data.describe())

#Replacing y,n to 1,0 Binary Data
voting_data.replace(('y','n'),(1,0), inplace=True)

#Replacing democrat and republican with 1,0
voting_data.replace(('democrat', 'republican'), (1, 0), inplace=True)

#getting all values except party type
all_features = voting_data[feature_names].drop('party',axis=1).values

#getting party type
all_classes = voting_data['party'].values

from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import cross_val_score

def create_model():

	# Model Sequential
	model = Sequential()

	# Layer with 32 neuron from weights and 16 inputs 
	model.add(Dense(32, input_dim=16,kernel_initializer='normal',activation='relu'))

	# Layer with 16 neuron and 1 output each
	model.add(Dense(16,kernel_initializer='normal',activation='relu'))

	#layer with 1 neuron
	model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))

	#Running it on a binary_crossentropy and rmsprop optimizer
	model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

	return model	

from keras.wrappers.scikit_learn import KerasClassifier

#Running create model with 100 epoch
estimator = KerasClassifier(build_fn=create_model, nb_epoch=100, verbose=0)

#using only 10 parties to predict stuffs
cv_scores = cross_val_score(estimator,all_features,all_classes,cv=10)

#printing it's mean
print(cv_scores.mean())
