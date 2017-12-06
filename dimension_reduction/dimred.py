import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import axes3d


df=pd.read_csv("/home/arindam/Desktop/tfmodel/data.csv")
in_data=df.drop(["Close","Volume"],axis=1)
label=df.drop(["Open","High","Low","Volume"],axis=1)
in_data["Open"] = pd.to_numeric(in_data["Open"],errors='coerce') #conversion of "Open" column to float so that it can be compared to "Close"
in_data.loc[:,("Result")]=in_data["Open"]<=label["Close"]
in_data.loc[:,("Result")]=in_data["Result"].astype(int)
pro=in_data.ix[in_data["Result"]==1]
loss=in_data.ix[in_data["Result"]==0]
in_data.loc[:,("Covar")]=in_data["Open"]<=label["Close"]

inputx=in_data.loc[:,["Open","High","Low"]].astype(np.float32).as_matrix()
pro_mat=pro.loc[:,["Open","High","Low"]].astype(np.float32).as_matrix()
loss_mat=loss.loc[:,["Open","High","Low"]].astype(np.float32).as_matrix()
inputy=in_data.loc[:,("Result")].as_matrix()
n=inputx.size/3

pro_x=pro_mat[:,0]
pro_y=pro_mat[:,1]
pro_z=pro_mat[:,2]

loss_x=loss_mat[:,0]
loss_y=loss_mat[:,1]
loss_z=loss_mat[:,2]
'''
fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")

ax.scatter(pro_x,pro_y,pro_z,c="b")
ax.scatter(loss_x,loss_y,loss_z,c="r")

ax.set_xlabel("Open")
ax.set_ylabel("High")
ax.set_zlabel("Low")

plt.show()
'''
covar=0
for m in inputx:
    mat=[]
    mat.append(m)
    mat=np.array(mat)
    covar+=(mat.T*mat)*(1/n)

u,s,v=np.linalg.svd(covar)
u_red=u[:,1:3]

input_x=[]
for i in inputx:
    m=[]
    m.append(i)
    m=np.array(m)
    input_x.append(np.matmul(m,u_red))

input_x=np.array(input_x)

pro_mat=[]
loss_mat=[]

for j in range(int(n)):
    if inputy[j]==1:
        pro_mat.append(input_x[j])
    else:
        loss_mat.append(input_x[j])

pro_x=[]
pro_y=[]
loss_x=[]
loss_y=[]

for p in pro_mat:
    pro_x.append(p[0][0])
    pro_y.append(p[0][1]) 

for l in loss_mat:
    loss_x.append(l[0][0])
    loss_y.append(l[0][1]) 

plt.scatter(pro_x,pro_y, color="blue", label="Profit")
plt.scatter(loss_x,loss_y, color="red", label="Loss")
    
plt.show()


'''
df=pd.read_csv("/home/arindam/Desktop/housedata.csv")

in_data=df.drop(["id","date","price","waterfront","view","yr_renovated","lat","long","sqft_living15","sqft_lot15"],axis=1)

# defining lambda functions
mult=lambda x: x*100
per5=lambda x: (x/5)*100
per11=lambda x :(x/10)*100
year=lambda x: (x-1900)
zipcode=lambda x: (x-98000)

in_data["bedrooms"]=in_data["bedrooms"].apply(lambda x : x*100)
in_data["bathrooms"]=in_data["bathrooms"].apply(mult)
in_data["floors"]=in_data["floors"].apply(mult)
in_data["condition"]=in_data["condition"].apply(per5)
in_data["grade"]=in_data["grade"].apply(per11)
in_data["yr_built"]=in_data["yr_built"].apply(year)
in_data["zipcode"]=in_data["zipcode"].apply(zipcode)

inputx=in_data.loc[0:1000,:].astype(np.float32).as_matrix()
n=inputx.size/(len(inputx[0]))

covar=0
for m in inputx:
    mat=[]
    mat.append(m)
    mat=np.array(mat)
    covar+=(mat.T*mat)*(1/n)

u,s,v=np.linalg.svd(covar)
u_red=u[:,0:3]

input_x=[]
for i in inputx:
    m=[]
    m.append(i)
    m=np.array(m)
    input_x.append(np.matmul(m,u_red))

input_x=np.array(input_x)
print(input_x)

pro_x=[]
pro_y=[]
pro_z=[]

for p in input_x:
    pro_x.append(p[0][0])
    pro_y.append(p[0][1]) 
    pro_z.append(p[0][2]) 

#plt.scatter(pro_x,pro_y, color="blue", label="Profit")
    
#plt.show()

#fig=plt.figure()
#ax=fig.add_subplot(111,projection="3d")

#ax.scatter(pro_x,pro_y,pro_z,c="b")

#plt.show()
'''




