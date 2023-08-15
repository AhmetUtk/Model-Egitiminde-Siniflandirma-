# Kütüphanelerin projeye dahil edilmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Veri setini okuma 
df = pd.read_csv('data.csv')

# İhtiyacımız olmayan kolonu silme
df=df.drop("Id",axis=1)

# Binary (İkili) Sınıflandırma 
df["quality"]=df['quality'].apply(lambda x:1 if x>6 else 0 )
df.to_csv("data.csv",index=False)



# quality kolonunu pasta grafiğine dökme
counts = df['quality'].value_counts()
fig, ax = plt.subplots()
ax.pie(counts, autopct='%1.1f%%')
ax.legend(labels=['0 (Normal)', '1 (Suspect)'], title='Dağılım',loc='lower right')
ax.set_title(" dağılım")
plt.show()
plt.close()


#Kolerasyon tapblosu 
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

ax, fig = plt.subplots(figsize=(15,15))
sns.heatmap(corr, vmin=-1, cmap='RdYlBu', annot=True, mask=mask)
plt.show()

corr_esik=0.20
x = df.drop(['quality'], axis=1)




# Polinomsal Dönüşüm
poly = pp.PolynomialFeatures(1)
x=poly.fit_transform(x)
x=pd.DataFrame(x)


# Hedef ve özelliklerin belirlenip ayrıldığı kısım
x = df.drop(['quality'], axis=1)
y = df['quality']



# Veri setini eğitim ve test için ayrıldığı kısım
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,shuffle=False, test_size=0.5)


# Modelin Oluşturulması
classifier = Sequential()


# Giriş Katmanının oluşturulması ve nöron ekleme 
l1=classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(x.columns), name="layer1"))

# Ara katmanlar oluşturup nöron ekleme
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', name="layer2"))
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))

# Çıkış katmanı oluşturma 
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Oluşturulan yapay sinir ağını derleyip eğitime başlatma kısmı
classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
classifier.fit(x_train, y_train, batch_size = 10, epochs = 2)

# Doğruluk değerine ulaşmak için 
score, acc = classifier.evaluate(x_train, y_train,batch_size=10)
print('Train score:', score)
print('Train accuracy:', acc)

# modelin tahmin etme kısmı
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

print('*'*20)
score, acc = classifier.evaluate(x_test, y_test,batch_size=10)

# Sonuçları ekrana yazdırma 
print('Test score:', score)
print('Test accuracy:', acc)


# Konfisyon Matris oluşumu
cm = confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print(classification_report(y_test,y_pred))


