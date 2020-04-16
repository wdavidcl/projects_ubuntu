import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D, LeakyReLU, Flatten, LSTM, SimpleRNN
import matplotlib.pyplot as plt
import pandas as pd

step = 1
steps_forward = 7

def convertToMatrix(y_k,u_k, step):
    X, Y =[], []
    for i in range(len(y_k)-step):
        d=i+step  
        X.append([y_k[i:d,],u_k[i:d,]])
        Y.append(y_k[d,])
    return np.array(X), np.array(Y)

data = pd.read_csv('./assets/covid.csv')
u = data.iloc[:, 0:1]
y = data.iloc[:, 1:2]
y_k = y.to_numpy(dtype = int)
u = u.to_numpy(dtype = 'datetime64')
u_k = np.arange(1,len(y)+1, dtype = int)

l_dataset = len(u_k)
l_train = int(1*l_dataset)

train = y_k
u_train = u_k

train = np.append(train,np.repeat(train[-1,],step))
u_train = np.append(u_train,np.repeat(u_train[-1,],step))
trainX,trainY = convertToMatrix(train,u_train,step)
print("shape:", trainX.shape)

model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(2,step), activation="relu"))
model.add(Dense(16, activation="relu")) 
model.add(Dense(8, activation="relu")) 
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.summary()

model.fit(trainX,trainY, epochs=1000, batch_size=1, verbose=2)
prediction = model.predict(trainX)
last_y = np.array([])
last_y = np.concatenate((last_y, prediction[-1]),axis=0)
u_final = u_test = np.arange(u_k[-1]+1,u_k[-1]+steps_forward) 

#print("original", u_test, u_test.shape)
for w in range(0,len(u_test)):
    u_test = np.append(u_test,np.repeat(u_test[-1,],step))
    #print("primera m:",u_test, u_test.shape )
    u = [prediction[-1],u_test[w]]
    testX = np.array(u)
    testX = testX.reshape(1, 2, step)
    new_y = model.predict(testX)
    prediction = np.concatenate((prediction,new_y),axis=0)
#print(u_k, u_test)
u_final = np.concatenate((u_k,u_final),axis =0)
plt.xlim(0,len(prediction))
plt.ylim(0,prediction[-1]+20)
plt.xlabel('Marzo 2020')
plt.ylabel('Casos confirmados Ecuador covid-19')
plt.grid()
plt.plot(u_k, y_k, '+') # plotting t, a separately 
plt.plot(u_final, prediction, '*','r') # plotting t, a separately 
#plt.axvline(y_k.index[l_train], c="y")
plt.savefig('./results/covid-19_EC_RNN.png')
plt.show()