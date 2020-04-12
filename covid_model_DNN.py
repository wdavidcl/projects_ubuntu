import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd

# Load training data
data = pd.read_csv('./assets/covid.csv')
u = data.iloc[:, 0:1]
y = data.iloc[:, 1:2]
y = y.to_numpy(dtype = int)
u = u.to_numpy(dtype = 'datetime64')
u = np.arange(1,len(y)+1, dtype = int)
#plt.xlim(0,30)
#plt.ylim(0,max(y))
plt.xlabel('(Mes)')
plt.ylabel('(Covid-19 casos confirmados)')
plt.plot(u,y, '*') # plotting t, a separately 
#plt.show()
days_ahead=15
u_pred = np.arange(1,len(y)+days_ahead, dtype= int)+1
df = pd.DataFrame(data = u_pred,columns=['Mes'])


# Define model
model = Sequential()
model.add(Dense(40, input_dim=1, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(u, y, epochs=5000, batch_size = 2)

predictions = model.predict(u_pred)

df2 = pd.DataFrame()
for i in range(0,len(predictions)):
    df2 = df2.append({"Casos Covid-19":int(predictions[i][0])},ignore_index = True)

df_out = pd.merge(df,df2,how = 'left',left_index = True, right_index = True)
df_out.to_csv('./results/DNN_covid_modelling.csv',index_label=False,index=False)

plt.grid()
plt.plot(u_pred, predictions, 'r','*') # plotting t, b separately 
plt.savefig('./results/DNN_graph.png')
plt.show()


