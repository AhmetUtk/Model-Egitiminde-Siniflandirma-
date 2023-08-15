# Kütüphaneleri projeye dahil etme 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dosya okuma
df = pd.read_csv('data1.csv')
# Gereksiz kolonu düşürme
df=df.drop("Id",axis=1)

# Multi (Çoklu) Sınıflandırma Fonksiyonu
def siniflandir(sayi):
    if sayi >=3 and sayi<5:
        return 0
    elif sayi >=5 and sayi <8:
        return 1
    else : 
        return 2

df["quality"]=df["quality"].apply(siniflandir)
df.to_csv("data1.csv",index=False)



counts = df['quality'].value_counts()
fig, ax = plt.subplots()
ax.pie(counts, autopct='%1.1f%%')
ax.legend(labels=['0 (Normal)', '1 (Suspect)'], title='Dağılım',loc='lower right')
ax.set_title(" dağılım")
plt.show()
plt.close()



corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

ax, fig = plt.subplots(figsize=(15,15))
sns.heatmap(corr, vmin=-1, cmap='RdYlBu', annot=True, mask=mask)
plt.show()

corr_esik=0.20

x = df.drop(['quality'], axis=1)


from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing as pp

poly = pp.PolynomialFeatures(1)
x=poly.fit_transform(x)
x=pd.DataFrame(x)


x = df.drop(['quality'], axis=1)
y = df['quality']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,shuffle=False, test_size=0.5)


from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
# Adding the input layer and the first hidden layer
l1=classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(x.columns), name="layer1"))

# Adding the second hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', name="layer2"))

# Adding the second hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN | means applying SGD on the whole ANN
classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

classifier.fit(x_train, y_train, batch_size = 10, epochs = 2)

score, acc = classifier.evaluate(x_train, y_train,
                            batch_size=10)
print('Train score:', score)
print('Train accuracy:', acc)
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

print('*'*20)
score, acc = classifier.evaluate(x_test, y_test,
                            batch_size=10)
print('Test score:', score)
print('Test accuracy:', acc)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

