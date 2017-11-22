import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dropout  
from sklearn.metrics import mean_squared_error as msr


df=pd.read_csv("/home/arindam/Desktop/tfmodel/csco.csv")
data=df.drop(["Close","Volume","High","Low","Date"],axis=1)

scale=MinMaxScaler(feature_range=(0,1))
data=scale.fit_transform(data)
train=data[0:150]
train=np.flip(train,0)
test=data[147:]
test=np.flip(test,0)

#plt.plot(data)
#plt.show()

look_back=3

X=[]
Y=[]
testX=[]

for i in range(look_back,len(train)):
    X.append(train[i-look_back:i])
    Y.append(train[i])

for i in range(look_back,len(test)):
    testX.append(test[i-look_back:i])

X=np.array(X)
Y=np.array(Y)
testX=np.array(testX)

model=Sequential()
model.add(LSTM(3,input_shape=(look_back,1)))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X,Y,epochs=350,batch_size=1,verbose=2)

predict=model.predict(X)
predict=np.flip(predict,0)
pred=model.predict(testX)
pred=np.flip(pred,0)
#predict=np.concatenate((predict,pred),axis=0)
plt.plot(data,color="blue")
plt.plot(predict,color="red")
plt.plot(list(range(149,len(pred)+149)),pred,color="m")
plt.show()







