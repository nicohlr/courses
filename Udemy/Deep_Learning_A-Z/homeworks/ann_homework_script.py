import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import keras
from keras.models import Sequential
from keras.layers import Dense

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Data preprocessing

path_train = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))), 'ressources/Artificial_Neural_Networks/Churn_Modelling.csv')
dataset = pd.read_csv(path_train)
print(dataset.head())
print(dataset.info())

X = dataset.iloc[:, 3:13].values # we modify indexes according to what we saw with the info() method of the dataset
y = dataset.iloc[:, 13].values # idem

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building our ANN

# Initializing our ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer of our ANN
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_shape = (11,)))
# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
# Compilling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Training the ANN
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # convert probabilities to binary output

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig = plt.subplot()
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual exited label')
plt.xlabel('Predicted exited label')
fig.xaxis.set_ticklabels(['Stay', 'Leave'])
fig.yaxis.set_ticklabels(['Stay', 'Leave']);
plt.show()

# K-fold cross validation

def build_ann():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_shape = (11,)))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_ann, batch_size=10, epochs=100, verbose=0)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=1)
print(accuracies)

mean = accuracies.mean()
variance = accuracies.std()
print('Mean accuracy:', mean, '\nVariance:', variance)

# Dropout Regularization

# Initializing our ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer of our ANN with dropout
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_shape = (11,)))
classifier.add(Dropout(p=0.1)) 
# Adding the second hidden layer with dropout
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))
# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
# Compilling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

