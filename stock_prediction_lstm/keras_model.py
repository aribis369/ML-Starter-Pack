from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.optimizers import SGD
import numpy as np
import pandas as pd

np.random.seed(5)
df=pd.read_csv("/home/arindam/Desktop/googl.csv")
in_data=df.drop(["Close","Volume"],axis=1)
label=df.drop(["Open","High","Low","Volume"],axis=1)
in_data.loc[:,("Result1")]=in_data["Open"]>=label["Close"]
in_data.loc[:,("Result1")]=in_data["Result1"].astype(int)
in_data.loc[:,("Result2")]=in_data["Open"]<label["Close"]
in_data.loc[:,("Result2")]=in_data["Result2"].astype(int)

inputx=in_data.loc[:,["Open","High","Low"]].astype(np.float32).as_matrix()
#print(inputx)
inputy=in_data.loc[:,("Result1")].as_matrix()
#print(inputy)

model=Sequential()
model.add(Dense(4,input_dim=3,init="uniform",activation="sigmoid"))
model.add(Dropout(0.2))
model.add(Dense(4,init="uniform",activation="relu"))
model.add(Dense(3,init="uniform",activation="relu"))
model.add(Dense(1,init="uniform",activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

model.fit(inputx, inputy, nb_epoch=50000, batch_size=50)

pred = model.predict(inputx)

print(pred)
    


