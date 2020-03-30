# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
from keras.models import model_from_json
import json
import numpy as np
import matplotlib.pyplot as plt

step = 1


with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('model_weights.h5')

model.summary()
# load dataset
u_test = 0*np.ones(200)
u_test = np.concatenate((u_test,5*np.ones(200)),axis=0)
u_test = np.concatenate((u_test,2.5*np.ones(200)),axis=0)
u_test = np.concatenate((u_test,1*np.ones(200)),axis=0)
u_test = np.concatenate((u_test,4.5*np.ones(200)),axis=0)
u_test = np.append(u_test,np.repeat(u_test[-1,],step))
#print("original", u_test, u_test.shape)
for w in range(0,len(u_test)):
    #print("primera m:",u_test, u_test.shape )
    if w == 0:
        u = [0,u_test[w]]
        testX = np.array(u)
        testX = testX.reshape(1, 2, step)
        prediction = model.predict(testX)
    else:
        u = [prediction[-1],u_test[w]]
        testX = np.array(u)
        testX = testX.reshape(1, 2, step)
        new_y = model.predict(testX)
        prediction = np.concatenate((prediction,new_y),axis=0)
    #print(len(u_test),len(index),len(prediction))
    #print(prediction)

try:
    plt.plot(np.arange(len(u_test)), u_test, 'b')
    plt.plot(np.arange(len(u_test)), prediction,'r')
finally:
    pass
plt.show()

#score = model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))