#python -m venv venv for creating a new virtual enviroment
import time
import ext_module
import pandas


#definition of parameters
T_MAX = 7
t = 0
reference = 4.527

#definition of functions
def controller(w,y):
  return -1.4048*(y)+2.4048*w

y=0
while t<T_MAX:

  u=controller(reference,y)
  y=ext_module.plant(u)
  print(y)
  t=0.1+t
  time.sleep(0.1)

ext_module.print('this works')
