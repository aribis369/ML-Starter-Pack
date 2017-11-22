import numpy as np
import pandas as pd
import math

df=pd.read_csv("/home/arindam/Desktop/collaborative-filtering/ratings.txt",sep=" ",header=None)
df.columns=["userid","movieid","rating"]

df=df.loc[df["movieid"]<=100]
inputx=df.loc[:,:].astype(np.float32).as_matrix()

mov=[]
c=0
for i in range(1,max(df["userid"])+1):
    mat=[]
    for j in range(c,len(df["userid"])):
        if i==inputx[j][0]:
            mat.append(np.array(inputx[j]))
        else:
            c=j
            break
 
    mov.append(np.array(mat))

mov=np.array(mov)

mov_mat=np.ones([100,max(df["userid"])])
mov_mat=mov_mat*6

for i in range(0,100):
    for m in mov:
        for r in m:
            if r[1]==i+1:
                mov_mat[i][int(r[0])-1]=r[2]
                break

mov_mat=np.array(mov_mat)
print(mov_mat.shape)
theta=[]
for i in range(max(df["userid"])):
    m=np.random.rand(1,15)
    theta.append(m)
theta=np.array(theta)
print(theta.shape)
feat=[]
for i in range(100):
    m=np.random.rand(15,1)
    feat.append(m)
feat=np.array(feat)
print(feat.shape)

epochs=3000
learn_rate=0.00035
lamda=0.0000001

#print((np.matmul(theta[0],feat[1]))[0][0])

for e in range(epochs):
    diff=0
    for r in range(100):
        for c in range(max(df["userid"])):
            if mov_mat[r][c]!=6:
                diff+=math.fabs((np.matmul(theta[c],feat[r]))[0][0]-mov_mat[r][c])
    print(diff)

    for r in range(100):
        diff=0
        for c in range(max(df["userid"])):
            if mov_mat[r][c]!=6:
                diff+=((np.matmul(theta[c],feat[r]))[0][0]-mov_mat[r][c])*(theta[c].T)
        feat[r]-=learn_rate*(diff+(lamda*feat[r]))

    for c in range(max(df["userid"])):
        diff=0
        for r in range(100):
            if mov_mat[r][c]!=6:
                diff+=((np.matmul(theta[c],feat[r]))[0][0]-mov_mat[r][c])*(feat[r].T)
        theta[c]-=learn_rate*(diff+(lamda*theta[c]))

print(np.matmul(theta[0],feat[0]))
print(np.matmul(theta[0],feat[1]))
print(np.matmul(theta[0],feat[2]))
print(np.matmul(theta[0],feat[3]))
print(np.matmul(theta[0],feat[4]))
print(np.matmul(theta[0],feat[5]))
print(np.matmul(theta[0],feat[6]))
print(np.matmul(theta[0],feat[7]))
print(np.matmul(theta[0],feat[8]))
print(np.matmul(theta[0],feat[9]))

