import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

############################## Data preprocessing ##############################

path_train = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__')))), 'ressources/Self_Organizing_Maps/Credit_Card_Applications.csv')
dataset = pd.read_csv(path_train)
dataset.head()

# Attributes
X = dataset.iloc[:, :-1].values
# Class (approved of not)
y = dataset.iloc[:, -1].values

# Features scaling (normalization)
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

############################## Building the SOM ##############################

# initializing som
som = MiniSom(x=10, y=10, input_len=15)
som.random_weights_init(X) # initializing weights randomly close to 0
som.train_random(data=X, num_iteration=100) # We don't train on X and y, just on X because it's unsueprvised learning

############################## Plotting the SOM ##############################

plt.figure(figsize=(20,10))
pcolor(som.distance_map().T) # distance_map method will return all the mean distances for winning nodes in one matrix
colorbar()

# add red circles and green squares for customer who did not get approval and customer who got approval respectively
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x) # get coordinate of winning node for observation x
    plot(w[0] + 0.5, 
         w[1] + 0.5, # plot marker AT THE MIDDLE of the square representing the winning node
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10, # graphical settings
         markeredgewidth=2) 

show()

############################## Finding frauds ##############################

mapping = som.win_map(X) # dict containing the observations pet node of the SOM
frauds = mapping[2, 4] # get customers in white winning node of SOM above
# frauds = np.concatenate((mapping[2, 4], mapping[, ]), axis=0) # to concatenate 2 nodes

frauds = sc.inverse_transform(frauds)
frauds = pd.DataFrame(frauds, columns=dataset.columns.tolist()[:-1])
frauds.head()

############################## Mega case study ##############################

df = dataset.copy()
df['Fraud'] = d.apply(lambda row: 1 if row.CustomerID in frauds['CustomerID'].astype(int).tolist() else 0, axis=1)

X2 = df.iloc[:, 1:16].values
y2 = df.iloc[:, 16].values

# Feature Scaling
sc = StandardScaler()
X2 = sc.fit_transform(X2)

classifier = Sequential()
classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu', input_shape = (15,)))
classifier.add(Dropout(p=0.1)) 
classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X2, y2, batch_size=10, epochs=100)

fraud_probability = classifier.predict(X2)
df['Fraud_proba'] = fraud_probability

df = df[['CustomerID', 'Fraud_proba']]
result = df.sort_values('Fraud_proba', axis=0, ascending= False)
result.head(50)