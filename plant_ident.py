import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D, LeakyReLU, Flatten, LSTM, SimpleRNN
import matplotlib.pyplot as plt
import pandas as pd
import json

step = 1

def convertToMatrix(y_k,u_k, step):
    X, Y =[], []
    for i in range(len(y_k)-step):
        d=i+step  
        X.append([y_k[i:d,],u_k[i:d,]])
        Y.append(y_k[d,])
    return np.array(X), np.array(Y)

data = pd.read_csv('heater.csv')
y_k = data.iloc[:, 0:1]
u_k = data.iloc[:, 1:2]

l_dataset = len(u_k)
l_train = int(0.8*l_dataset)

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
model.add(Dense(16, activation="relu")) 
model.add(Dense(8, activation="relu")) 
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.summary()

model.fit(trainX,trainY, epochs=500, batch_size=8, verbose=2)
prediction = model.predict(trainX)

test = np.append(test,np.repeat(test[-1,],step))
u_test = np.append(u_test,np.repeat(u_test[-1,],step))
testX,testY = convertToMatrix(test,u_test,step)
prediction_test = model.predict(testX)
prediction = np.concatenate((prediction,prediction_test),axis=0)


index = y_k.index.values
#plt.xlim(0,len(prediction))
#plt.ylim(0,15)
plt.xlabel('Eingang: u')
plt.ylabel('Ausgang: y')
plt.grid()
plt.plot(index, u_k, 'g') # plotting t, a separately 
plt.plot(index, y_k, '*')
plt.plot(index, prediction, 'r')

offset = len(index)
u_test = 0*np.ones(200)
u_test = np.concatenate((u_test,5*np.ones(200)),axis=0)
u_test = np.concatenate((u_test,2.5*np.ones(200)),axis=0)
u_test = np.concatenate((u_test,1*np.ones(200)),axis=0)
u_test = np.concatenate((u_test,4.5*np.ones(200)),axis=0)
u_test = np.append(u_test,np.repeat(u_test[-1,],step))
#print("original", u_test, u_test.shape)
for w in range(0,len(u_test)):
    #print("primera m:",u_test, u_test.shape )
    u = [prediction[-1],u_test[w]]
    testX = np.array(u)
    testX = testX.reshape(1, 2, step)
    new_y = model.predict(testX)
    prediction = np.concatenate((prediction,new_y),axis=0)
    #print(len(u_test),len(index),len(prediction))

try:
    plt.plot(np.arange(len(u_test))+offset, u_test, 'b')
    plt.plot(np.arange(len(u_test))+offset, prediction[offset:],'r')
finally:
    pass
plt.show()

# serialize model to JSON
model_json = model.to_json()
with open("model_in_json.json", "w") as json_file:
    json.dump(model_json, json_file)

model.save_weights("model_weights.h5")