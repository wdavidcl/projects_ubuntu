import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.read_csv('plant_samples.csv')

u = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],  dtype=int)
y = np.array([5,6,7,10,13,13,14,15,15,17,17,19,21,23,37,58,111],  dtype=float)

"""for i,c in enumerate(u):
  print("{} volts = {} RPM".format(c, y[i]))"""

plt.scatter(u, y)
plt.xlim(0,30)
plt.ylim(0,3000)
plt.xlabel('(Marzo)')
plt.ylabel('(Covid-19 casos confirmados)')
plt.show() 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(u,y,random_state=42,train_size=0.8, test_size=0.2)

l_0 = tf.keras.layers.Dense(units=6, input_shape=[1])
l_1 = tf.keras.layers.Dense(units=7)
l_2 = tf.keras.layers.Dense(units=1) 
model = tf.keras.Sequential([l_0, l_1, l_2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))

""" 
layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer_0])
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1)) """


trained_model = model.fit(X_train, y_train, epochs=30000, verbose=False)
print("Finished training the model")

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(trained_model.history['loss'])
plt.show()
print(model.predict([18]))
print(model.predict([19]))
print(model.predict([20]))
print(model.predict([21]))

y_pred = model.predict(X_test)
print('Actual Values\tPredicted Values')
print(y_test,'   ',y_pred.reshape(1,-1))

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)