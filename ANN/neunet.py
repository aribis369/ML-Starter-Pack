import numpy as np
import pandas as pd
from decimal import *


in_vec=[]
out_vec=[]
ni=int()
no=int()
nodes=[]
n_nodes=int()
theta=[]
error=[]
activ=[]
e=float()
error=[]
deriv=[]
learning_rate=float()
norm=float()
cycles=int()

def forw_prop():
    global nodes
    global in_vec
    global theta
    global activ
    global e
    for i in range(len(nodes)-1):
        activ[i+1]=np.matmul(theta[i],activ[i])
        activ[i+1]=(1/(1+(float(e)**(-activ[i+1]))))


def back_prop():
    global theta
    global activ
    global in_vec
    global out_vec
    global error
    error[len(nodes)-2]=np.subtract(activ[-1],out_vec)
    for i in reversed(range(len(nodes)-2)):
        error[i]=(activ[i+1])*(np.subtract(np.ones((nodes[i+1],1)),activ[i+1]))

    print(error)
 
 
def update_weight():
    global theta
    global activ
    global nodes
    global out_vec
    global error
    global deriv
    global learning_rate
    global norm
    global cycles
    for i in range(len(nodes)-1):
        d=np.matmul(error[i],activ[i].T)
        deriv.append(d)
    dervi=np.array(deriv)
    print("deriv",deriv)

    for c in range(cycles):
        #theta=np.subtract(theta,(learning_rate*deriv))
        theta=np.subtract(theta,(np.multiply(deriv,learning_rate)))
        #print(c+1)
    print("New theta",theta)
    print("New activ")
    forw_prop()
    print(activ)
    
            

def input_rand():
    global ni
    global no
    global n_nodes
    global nodes
    global in_vec
    global theta
    global activ
    global e
    global in_vec
    global out_vec
    global error
    global learning_rate
    global cycles
    e=Decimal(1).exp()
    getcontext().prec = 40
    ni=int(input("Enter the number of elements in input vector = "))
    no=int(input("Enter the number of elements in output vector = "))
    n_nodes=int(input("Enter the number of hidden layers = "))

 
    nodes.append(ni)
    for i in range(n_nodes):
        u=int(input("Enter number of nodes in "+str(i+1)+" hidden layer = "))
        nodes.append(u)
    nodes.append(no)

    learning_rate=float(input("Enter the learing rate = "))
    cycles=int(input("Enter the no. of cycles = "))

    #place the data cleaning and input & output part here(input data to in_vec & out_vec)
    out_vec=[[1],[1],[0],[1],[1],[0],[0],[1]]
    out_vec=np.array(out_vec)

    

    for j in range(len(nodes)-1):
        t=np.random.rand(nodes[j+1],nodes[j])
        theta.append(t)
   
    theta=np.array(theta)
    print("theta",theta)
 
    for l in range(len(nodes)-1):
        z=np.ones((nodes[l],1))
        activ.append(z)

    activ.append(out_vec)
    activ=np.array(activ)
    print("first activ",activ)

    forw_prop()

    for l in range(len(nodes)-1):
        er=np.zeros((nodes[l],1))
        error.append(er)
 
    back_prop()
    update_weight()

    

input_rand()
