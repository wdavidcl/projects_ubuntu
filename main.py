#python -m venv venv for creating a new virtual enviroment
import time
import ext_module
import pandas as pd
import tensorflow as tf
#definition of parameters
T_MAX = 10
reference = [2,7,13,4,9,1]
#definition of functions
def controller(w,y):
  return -1.4048*(y)+2.4048*w

y=0
samples = pd.DataFrame({})
for values in reference:
  t = 0
  while t<T_MAX:
    #u = controller(values,y)
    x = ext_module.plant(values)
    y = x
    print(y)
    samples=samples.append({"u":values,"y":y},ignore_index = True)
    t = 0.1+t
    time.sleep(0.1)

ext_module.print('this works')
print(samples)
samples.to_csv('plant_samples.csv',index_label=False, header=None,index=False)
