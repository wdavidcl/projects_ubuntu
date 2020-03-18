import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D, LeakyReLU, Flatten, LSTM
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('plant_samples.csv')
u = data.iloc[:, 0:1]
y = data.iloc[:, 1:2]

limit = len(u)
t = np.arange(limit)+1

plt.xlim(0,len(t))
plt.ylim(0,15)
plt.xlabel('Eingang: u')
plt.ylabel('Ausgang: y')
plt.grid()
plt.plot(t, u, 'r') # plotting t, a separately 
plt.plot(t, y, 'b') # plotting t, a separately 
#plt.show()
print(u.shape[0],u.shape[1])
u = np.reshape(u, (u.shape[0], 1, u.shape[1]))

model = Sequential()
model.add(LSTM(32, return_sequences=True,input_shape=(limit, 1)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(u, y,batch_size=101, epochs=5)

y_ml = model.predict(u)

plt.plot(t, y_ml, 'y') # plotting t, a separately 
plt.show()