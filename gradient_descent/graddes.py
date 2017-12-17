import numpy as np
import pandas as pd
import math
import datetime
from sklearn.utils import shuffle

df=pd.read_csv("G:\Kharagpur Studies\GSOC\KWOC\ML Starter Pack\ML-Starter-Pack\gradient_descent\data.csv")
in_data=df.drop(["Close","Volume"],axis=1)
label=df.drop(["Open","High","Low","Volume"],axis=1)

in_data.loc[:,("Result1")]=in_data["Open"]>=label["Close"]
in_data.loc[:,("Result1")]=in_data["Result1"].astype(int)
in_data.loc[:,("Result2")]=in_data["Open"]<label["Close"]
in_data.loc[:,("Result2")]=in_data["Result2"].astype(int)


def norm_grad_des():
    global in_data
    inputx=in_data.loc[0:1000,["Open","High","Low"]].astype(np.float32).as_matrix()
    print(inputx.shape)
    inputy=in_data.loc[0:1000,["Result1","Result2"]].as_matrix()
    print(inputy.shape)
    
    n=len(inputy/2)
    w=np.random.rand(3,2)*0.01
    learn_rate=0.000001
    epochs=5000
    lamda=0.0001
 
    X=[]
    Y=[]
  
    for i in range(n):
        x=[]
        y=[]
        x.append(inputx[i])
        x=np.array(x)
        X.append(x)
        y.append(inputy[i])
        y=np.array(y)
        Y.append(y)
 
    inputx=np.array(X)
    inputy=np.array(Y)
    print(inputy)
    ti=datetime.datetime.now()

    for e in range(epochs):
        der=np.zeros([3,2])
        err=0.0

        for i in range(n):
            r=(1/(1+pow(math.e,-np.matmul(inputx[i],w))))
            '''
            print(pow(math.e,-np.matmul(inputx[i],w)))
            print(r[0])
            '''
            err+=(-1/n)*(inputy[i][0][1]*math.log(r[0][1])+((1-inputy[i][0][1])*math.log(1-r[0][1])))
            der+=(1/n)*np.matmul(inputx[i].T,((1/(1+pow(math.e,-np.matmul(inputx[i],w))))-inputy[i]))
        e=w**2
        s=e.sum()
        print(err+(lamda/(2*n))*s)
        w-=learn_rate*(der+(lamda/(n))*w)

    t=datetime.datetime.now()-ti
    print("time required = ",t)

    #print(w)
    for i in range(15):
        print(np.matmul(inputx[i],w)[0])
        print(inputy[i][0])


#def stochastic_grad_des():
#    global in_data
#    in_data=shuffle(in_data)
#    '''
#    inputx=in_data.loc[:,["Open","High","Low"]].astype(np.float32).as_matrix()
#    inputy=in_data.loc[:,["Result1","Result2"]].as_matrix()
#    '''
#    n=len(in_data)
#    w=np.ones([3,2])*0.001
#    learn_rate=0.001
#    epochs=43
#    lamda=0.0001
#    '''
#    X=[]
#    Y=[]
#  
#    for i in range(n):
#        x=[]
#        y=[]
#        x.append(inputx[i])
#        x=np.array(x)
#        X.append(x)
#        y.append(inputy[i])
#        y=np.array(y)
#        Y.append(y)
# 
#    inputx=np.array(X)
#    inputy=np.array(Y)
#    print(inputy)
#    ti=datetime.datetime.now()
#    '''
#
#    for e in range(epochs):
#        in_data=shuffle(in_data)
#        inputx=in_data.loc[:,["Open","High","Low"]].astype(np.float32).as_matrix()
#        inputy=in_data.loc[:,["Result1","Result2"]].as_matrix()
# 
#        X=[]
#        Y=[]
#  
#        for i in range(n):
#            x=[]
#            y=[]
#            x.append(inputx[i])
#            x=np.array(x)
#            X.append(x)
#            y.append(inputy[i])
#            y=np.array(y)
#            Y.append(y)
# 
#        inputx=np.array(X)
#        inputy=np.array(Y)
#
#        err=0.0
#        for k in range(n):
#            r=(1/(1+pow(math.e,-np.matmul(inputx[i],w))))
#            err=(inputy[i][0][1]*math.log(r[0][1])+((1-inputy[i][0][1])*math.log(1-r[0][1])))
#            der=(np.matmul(inputx[k].T,((1/(1+pow(math.e,-np.matmul(inputx[k],w))))))-inputy[k])
#            w-=learn_rate*(der+((lamda)*w))
#           
#        print(err)
#        #    err=0.0
#        
#    t=datetime.datetime.now()-ti
#    print("time required = ",t)
#
#    for i in range(5):
#        print(np.matmul(inputx[i],w)[0])
#        print(inputy[i][0])
#    

def mini_batch_grad_des():
    global in_data
    inputx=in_data.loc[0:1000,["Open","High","Low"]].astype(np.float32).as_matrix()
    print(inputx.shape)
    inputy=in_data.loc[0:1000,["Result1","Result2"]].as_matrix()
    print(inputy.shape)
    
    n=len(inputy/2)
    w=np.random.rand(3,2)*0.01
    learn_rate=0.000001
    epochs=25
    lamda=0.0001
    b=10
 
    X=[]
    Y=[]
  
    for i in range(n):
        x=[]
        y=[]
        x.append(inputx[i])
        x=np.array(x)
        X.append(x)
        y.append(inputy[i])
        y=np.array(y)
        Y.append(y)
 
    inputx=np.array(X)
    inputy=np.array(Y)
    print(inputy)
    ti=datetime.datetime.now()

    for e in range(epochs):

        for j in range(0,n-b,b):
            
            der=np.zeros([3,2])
            err=0.0
            
            for i in range(j,j+b):
                r=(1/(1+pow(math.e,-np.matmul(inputx[i],w))))
                '''
                print(pow(math.e,-np.matmul(inputx[i],w)))
                print(r[0])
                '''
                err+=(-1/b)*(inputy[i][0][1]*math.log(r[0][1])+((1-inputy[i][0][1])*math.log(1-r[0][1])))
                der+=(1/b)*np.matmul(inputx[i].T,((1/(1+pow(math.e,-np.matmul(inputx[i],w))))-inputy[i]))
                
            e=w**2
            s=e.sum()
            print(err+(lamda/(2*b))*s)
            w-=learn_rate*(der+(lamda/(b))*w)
            

    t=datetime.datetime.now()-ti
    print("time required = ",t)

    #print(w)
    for i in range(15):
        print(np.matmul(inputx[i],w)[0])
        print(inputy[i][0])

   
#norm_grad_des()        
#stochastic_grad_des()
mini_batch_grad_des()
