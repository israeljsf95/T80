
#Israel: Xor model using Keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
import numpy as np 


def carrega_data():
  file = open('../dados/Tabela5.9.txt',)
  x = np.loadtxt(file, skiprows = 1, usecols = (0, 1, 2, 3,))
  x = np.array(x)
  file.seek(0,0)
  y = np.loadtxt(file, skiprows = 1, usecols=(4, 5, 6,))
  y = np.array(y)
  file = open('../dados/Tabela5.9t.txt',)
  tx = np.loadtxt(file, skiprows = 1, usecols = (0,1,2,3,))
  tx = np.array(tx)
  file.seek(0,0)
  ty = np.loadtxt(file, skiprows = 1, usecols=(4, 5, 6,))
  ty = np.array(ty)
  return (x, y, tx, ty)




x, y, x_t, y_t = carrega_data()


model = Sequential()
model.add(Dense(15, input_dim=4, init = 'lecun_uniform', activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
sgd = SGD(lr = 0.1, momentum = 0.9)
model.compile( loss = 'categorical_crossentropy',
			   optimizer = 'sgd',
			   metrics = ['accuracy'])

result = model.fit(x, y, nb_epoch = 1000, verbose = 2)
print(result.history.keys())


print (model.predict(x).round())
print (model.predict(x_t).round())
# summarize history for accuracy
plt.plot(result.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend('train', loc='upper left')
plt.show()
# summarize history for loss
plt.plot(result.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend('train', loc='upper left')
plt.show()
