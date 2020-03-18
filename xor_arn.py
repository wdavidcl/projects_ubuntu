import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd

# Load training data
a = np.array([1,1,0,0],  dtype=int)
b = np.array([1,0,1,0],  dtype=int)
y = np.array([0,1,1,0],  dtype=float)
"""plt.xlim(0,30)
plt.ylim(0,1500)
plt.xlabel('(Marzo)')
plt.ylabel('(Covid-19 casos confirmados)')
plt.grid()
plt.plot(u, y, 'b') # plotting t, a separately 
u_pred = np.arange(30)+1
df = pd.DataFrame(data = u_pred,columns=['Marzo'])"""

u = np.column_stack((a,b))

# Define model
model = Sequential()
model.add(Dense(40, input_dim=2, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(u, y, epochs=300, batch_size=1)

predictions = model.predict(u)

for i in range(0,len(predictions)):
    print(int((predictions[i])))
