import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data2.csv')


df.sample(10)

df = df.drop(['Id'], axis=1)

from keras.utils import to_categorical


x = df.drop(['quality'], axis=1)
y = df['quality']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,shuffle=False, test_size=0.25)

y_train= to_categorical(y_train)
y_test= to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
# Adding the input layer and the first hidden layer
l1=classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(x.columns), name="layer1"))

# Adding the second hidden layer
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', name="layer2"))

# Adding the second hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(9, activation='relu'))

# Compiling the ANN | means applying SGD on the whole ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(x_train, y_train, batch_size = 10, epochs = 128)

score, acc = classifier.evaluate(x_train, y_train,batch_size=10)
print('Train score:', score)
print('Train accuracy:', acc)
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x_test)

print('*'*20)
score, acc = classifier.evaluate(x_test, y_test,batch_size=10)
print('Test score:', score)
print('Test accuracy:', acc)
# Making the Confusion Matrix