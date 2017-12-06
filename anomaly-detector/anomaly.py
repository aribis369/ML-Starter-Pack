import numpy as np
import pandas as pd
import math

df=pd.read_csv("/home/arindam/Desktop/anomaly-detector/sat_data.csv")

inputx=df.loc[:,["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36"]].astype(np.float32).as_matrix()
    
inputy=df.loc[:,("37")].as_matrix()

outlier=[]
normal=[]

for i in range(len(inputy)):
    if inputy[i]=="o":
        outlier.append(inputx[i])
    else:
        normal.append(inputx[i])

# 75 vectors in outlier and 5075 vectors in normal
outlier=np.array(outlier)
normal=np.array(normal)

# n=36 i.e. no. of features
# mean vector(1xn)
mean=normal.mean(0)

#print(mean)
# cal dimension of input vector(n=36)
n=len(inputx[0])
# no.of input vectors
num=len(normal)
#print(num)

'''
mat=[]
mat.append(np.subtract(normal[0],mean))
mat=np.array(mat)
'''
#print(mat)
#print(mat.T)

# cal covariance matrix
covar=0

for m in normal:
    mat=[]
    mat.append(np.subtract(m,mean))
    # dimen of mat = (1xn)
    mat=np.array(mat)
    # dimen of covar = (nxn)
    covar+=(mat.T*mat)*(1/num)

# taking transpose of covar as all "mat" vectors are (1xn) but as per convention they should be (nx1)
# taking tanspose will help us get the original covariance matrix 
covar=covar.T/1e2
#print(covar)
sigma=np.linalg.det(covar)
#x=outlier[100]
print("sqrt",math.sqrt(sigma))

res=[]

for r in normal:
    w=[]
    w.append(np.subtract(r,mean))
    # dimen of w = (1xn)
    w=np.array(w)
    #print(np.shape(w))
    u=np.matmul((np.matmul(w,np.linalg.inv(covar))),w.T)
    #print(w)
    #print(u[0][0])
    #print(np.linalg.inv(covar))
    #print(np.matmul((np.subtract(r,mean)),np.linalg.inv(covar)))
    #print(w.T)
    #print(u)
    #prob=(1/(pow((2*math.pi),(n/2))*math.sqrt(sigma)))*(pow(math.e,((-0.5)*u)))
    '''
    print(u[0][0])
    print(pow((2*math.pi),(n/2))*math.sqrt(sigma))
    print((1e2*pow(math.e,((-0.5)*(u[0][0]/1e3)))))
    print((1/(pow((2*math.pi),(n/2))*math.sqrt(sigma))))
    '''
    prob=(1/(pow((2*math.pi),(n/2))*math.sqrt(sigma)))*(1e2*pow(math.e,((-0.5)*(u[0][0]/1e3))))
    #prob=(1/(pow((2*math.pi),(n/2))*math.sqrt(sigma)))*(pow(math.e,((-0.5)*u[0][0])/1e30))
    print(prob)
    # probability greater than min. prob i.e. 5e-26(arbit.)
    if prob>=(40):
        res.append(1)

Res=[]

for r in outliner:
    w=[]
    w.append(np.subtract(r,mean))
    # dimen of w = (1xn)
    w=np.array(w)
    #print(np.shape(w))
    u=np.matmul((np.matmul(w,np.linalg.inv(covar))),w.T)
    #print(w)
    #print(u[0][0])
    #print(np.linalg.inv(covar))
    #print(np.matmul((np.subtract(r,mean)),np.linalg.inv(covar)))
    #print(w.T)
    #print(u)
    #prob=(1/(pow((2*math.pi),(n/2))*math.sqrt(sigma)))*(pow(math.e,((-0.5)*u)))
    '''
    print(u[0][0])
    print(pow((2*math.pi),(n/2))*math.sqrt(sigma))
    print((1e2*pow(math.e,((-0.5)*(u[0][0]/1e3)))))
    print((1/(pow((2*math.pi),(n/2))*math.sqrt(sigma))))
    '''
    prob=(1/(pow((2*math.pi),(n/2))*math.sqrt(sigma)))*(1e2*pow(math.e,((-0.5)*(u[0][0]/1e3))))
    #prob=(1/(pow((2*math.pi),(n/2))*math.sqrt(sigma)))*(pow(math.e,((-0.5)*u[0][0])/1e30))
    print(prob)
    # probability greater than min. prob i.e. 5e-26(arbit.)
    if prob<(40):
        Res.append(1)

print("No. of normal vectors (predicted) = ",len(res))

print("Percentage of correct predictions = ",(len(res)/len(normal))*100)

print("No. of outliner vectors (predicted) = ",len(Res))

print("Percentage of correct predictions = ",(len(Res)/len(outliner))*100)

'''
res=[]
for r in normal:
    w=[]
    w.append(np.subtract(r,mean))
    w=np.array(w)
    u=np.matmul((np.matmul(w,np.linalg.inv(covar))),w.T)
    #u=np.matmul((np.matmul((np.subtract(r,mean)),np.linalg.inv(covar))),(np.subtract(r,mean).T))
    #print(u)
    #prob=(1/(pow((2*math.pi),(n/2))*math.sqrt(sigma)))*(pow(math.e,((-0.5)*u)))
    prob=(1/(pow((2*math.pi),(n/2))))*(pow(math.e,((-0.5)*u)))
    print(prob)
    if prob>(5e-26):
        res.append(1)
print(len(res))
print((len(res)/len(normal))*100)
'''
#print(mean)

