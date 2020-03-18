import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd

# Load training data
u = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],  dtype=int)
y = np.array([5,6,7,10,13,14,15,16,17,17,18,19,21,23,37,58,111,168],  dtype=float)
plt.xlim(0,30)
plt.ylim(0,1500)
plt.xlabel('(Marzo)')
plt.ylabel('(Covid-19 casos confirmados)')
plt.grid()
plt.plot(u, y, 'b') # plotting t, a separately 
u_pred = np.arange(30)+1
df = pd.DataFrame(data = u_pred,columns=['Marzo'])


# Define model
model = Sequential()
model.add(Dense(40, input_dim=1, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(u, y, epochs=3000, batch_size=2)

predictions = model.predict(u_pred)
df2 = pd.DataFrame()
for i in range(0,len(predictions)):
    df2 = df2.append({"Casos Covid-19":int(predictions[i][0])},ignore_index = True)

df_out = pd.merge(df,df2,how = 'left',left_index = True, right_index = True)
df_out.to_csv('covid.csv',index_label=False,index=False)

plt.plot(u_pred, predictions, 'r') # plotting t, b separately 
plt.show()


