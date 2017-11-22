import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random
from scipy.spatial import distance

def error(r,l):
    e=0.0
    for i in range(len(inputx)):
        e+=distance.euclidean(inputx[i],r[l[i]])

    return(e)
    


df=pd.read_csv("/home/arindam/Desktop/KNN algo/Iris.csv")

inputx=df.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]].astype(np.float32).as_matrix()
inputy=df.loc[:,("Species")].as_matrix()

for i in range(len(inputy)):
    if inputy[i]=="Iris-setosa":
        inputy[i]=0
    elif inputy[i]=="Iris-versicolor":
        inputy[i]=1
    else:
        inputy[i]=2

min_mat=[]
min_err=[]

for n in range(2,9):
    rand=[]
    print("val n = ",n)
    mat=[]
    minerr=0
    for r in range(10):
        rand=[]
        rd=random.sample(range(0,150),n)
        for j in rd:
            rand.append(inputx[j])

        for nt in range(100):
            label=[]
            for i in inputx:
                dis=[]
                for j in range(n):
                    dis.append(distance.euclidean(i,rand[j]))
                label.append(dis.index(min(dis)))

            for k in range(n):
                emp=[]
                for l in range(len(inputx)):
                    if label[l]==k:
                        emp.append(inputx[l])
                emp=np.array(emp)
                centroid=emp.mean(0)
                rand[k]=centroid
        
        #print("rand = ",rand)

        label=[]
        for i in inputx:
            dis=[]
            for j in range(n):
                dis.append(distance.euclidean(i,rand[j]))
            label.append(dis.index(min(dis)))

        print(error(rand,label))    
        if r==0:        
            mat.append(rand)
            mat.append(label)
            minerr=error(rand,label)
        else:
            if error(rand,label)<minerr:
                mat=[]
                mat.append(rand)
                mat.append(label)
                minerr=error(rand,label)
   
    print("Minimum = ",minerr)
    print("Mat = ",mat)
    min_err.append(minerr)
    temp=[]
    temp.append(mat[0])
    temp.append(mat[1]) 
    min_mat.append(temp)   
'''
    if n==2:        
        min_mat.append(mat[0])
        min_mat.append(mat[1])
        minerr=error(mat[0],mat[1])
    else:
        if error(rand,label)<minerr:
            min_mat=[]
            min_mat.append(mat[0])
            min_matmat.append(mat[1])
            minerr=error(mat[0],mat[1])
'''
   
print("Minimum = ",minerr)
print("Min_mat = ",min_mat)
print("Min_err = ",min_err)

plt.scatter(range(2,len(min_err)+2),min_err, color="blue", label="Iris")

plt.show()
            
            
           


