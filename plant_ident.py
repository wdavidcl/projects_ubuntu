import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D, LeakyReLU, Flatten, LSTM, SimpleRNN
import matplotlib.pyplot as plt
import pandas as pd

step = 1

def convertToMatrix(y_k,u_k, step):
    X, Y =[], []
    for i in range(len(y_k)-step):
        d=i+step  
        X.append([y_k[i:d,],u_k[i:d,]])
        Y.append(y_k[d,])
    return np.array(X), np.array(Y)

data = pd.read_csv('plant_samples.csv')
u_k = data.iloc[:, 0:1]
y_k = data.iloc[:, 1:2]

l_dataset = len(u_k)
l_train = int(1*l_dataset)

values = y_k.values
train,test = values[0:l_train,:], values[l_train:l_dataset,:]
#escribir aqui este deseado
values = u_k.values
u_train,u_test = values[0:l_train,:], values[l_train:l_dataset,:]

train = np.append(train,np.repeat(train[-1,],step))
u_train = np.append(u_train,np.repeat(u_train[-1,],step))
trainX,trainY = convertToMatrix(train,u_train,step)
print("shape:", trainX.shape)

model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(2,step), activation="relu"))
model.add(Dense(8, activation="relu")) 
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.summary()

model.fit(trainX,trainY, epochs=100, batch_size=10, verbose=1)
prediction = model.predict(trainX)

u_test = 1*np.ones(51)
u_test = np.concatenate((u_test,np.linspace(1.0, 14.23, num=101)),axis=0)
u_test = np.concatenate((u_test,5.6*np.ones(201)),axis=0)
u_test = np.concatenate((u_test,5.1*np.ones(201)),axis=0)
u_test = np.concatenate((u_test,1.87*np.ones(101)),axis=0)
#print("original", u_test, u_test.shape)
for w in range(0,len(u_test)):
    u_test = np.append(u_test,np.repeat(u_test[-1,],step))
    #print("primera m:",u_test, u_test.shape )
    u = [prediction[-1],u_test[w]]
    testX = np.array(u)
    testX = testX.reshape(1, 2, step)
    new_y = model.predict(testX)
    prediction = np.concatenate((prediction,new_y),axis=0)

index = y_k.index.values
plt.xlim(0,len(prediction))
plt.ylim(0,15)
plt.xlabel('Eingang: u')
plt.ylabel('Ausgang: y')
plt.grid()
plt.plot(index, u_k, 'g') # plotting t, a separately 
plt.plot(index, y_k, '*')
plt.plot(np.arange(len(prediction))+1, prediction, 'r')
plt.plot(np.arange(len(u_test))+len(index), u_test, 'g')
#plt.axvline(y_k.index[l_train], c="y")
plt.show()
