import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


df=pd.read_csv("/home/arindam/Desktop/googl.csv")
in_data=df.drop(["Close","Volume"],axis=1)
label=df.drop(["Open","High","Low","Volume"],axis=1)
in_data.loc[:,("Result1")]=in_data["Open"]>=label["Close"]
in_data.loc[:,("Result1")]=in_data["Result1"].astype(int)
in_data.loc[:,("Result2")]=in_data["Open"]<label["Close"]
in_data.loc[:,("Result2")]=in_data["Result2"].astype(int)

inputx=in_data.loc[:,["Open","High","Low"]].astype(np.float32).as_matrix()
print(inputx)
inputy=in_data.loc[:,["Result1","Result2"]].as_matrix()
print(inputy)

dfr=pd.read_csv("/home/arindam/Desktop/tfmodel/intc.csv")
in_data_r=dfr.drop(["Close","Volume"],axis=1)
label_r=dfr.drop(["Open","High","Low","Volume"],axis=1)
in_data_r.loc[:,("Result1")]=in_data_r["Open"]>=label_r["Close"]
in_data_r.loc[:,("Result1")]=in_data_r["Result1"].astype(int)

inputx_r=in_data_r.loc[:,["Open","High","Low"]].astype(np.float32).as_matrix()
print(inputx_r)
inputy_r=in_data_r.loc[:,("Result1")].as_matrix()
print(inputy_r)

learning_rate=0.0001
training_epochs=2000000
display_step=50
n=inputx.size/3
nr=inputx_r.size/3
cal_mat=[]

x=tf.placeholder(tf.float32,[None,3])
w=tf.Variable(tf.zeros([3,2]))
#b=tf.Variable(tf.zeros([1,2]))

#w=tf.Variable(tf.random_uniform([3,2]))
b=tf.Variable(tf.random_uniform([1,2]))

y_values=tf.add(tf.matmul(x,w),b)

y_val=tf.nn.softmax(y_values)
y=tf.placeholder(tf.float32,[None,2])

cost=tf.reduce_sum(tf.pow((y-y_val),2))/(2*n)
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

#can use the commented block too

for i in range(int(training_epochs)):
    if i<500000:
        sess.run(optimizer,feed_dict={x:inputx,y:inputy})
        if (i%50000)==0:
            print("Cost value = ",sess.run(cost,feed_dict={x:inputx,y:inputy}))
    elif i==500000:
        learning_rate*=30
        print("Learning Rate Increased")
    elif i>500000 and i<1000000:
        sess.run(optimizer,feed_dict={x:inputx,y:inputy})
        if (i%50000)==0:
            print("Cost value = ",sess.run(cost,feed_dict={x:inputx,y:inputy}))
    elif i==1000000:
        learning_rate*=30
        print("Learning Rate Increased")
    else:
        sess.run(optimizer,feed_dict={x:inputx,y:inputy})
        if (i%50000)==0:
            print("Cost value = ",sess.run(cost,feed_dict={x:inputx,y:inputy})) 
'''

cost_prev=0
cost_val=0

for i in range(int(training_epochs)):
    if (i%50000)==0 and i!=0:
        #sess.run(optimizer,feed_dict={x:inputx,y:inputy})
        cost_prev=cost_val
        cost_val = sess.run(cost,feed_dict={x:inputx,y:inputy})
        print(cost_val)
        if cost_prev-cost_val<0.001 and cost_prev-cost_val>0.0005 :
            learning_rate*=30
            print("Learning Rate Increased")
        elif cost_prev-cost_val<0.0005:
            learning_rate*=900
            print("Learning Rate Increased twice")
    else:
        sess.run(optimizer,feed_dict={x:inputx,y:inputy})

'''    
print("Finished optimizing")
print(sess.run(w))
print(sess.run(y_val,feed_dict={x:inputx}))
mat=sess.run(y_val,feed_dict={x:inputx})

for i in range(int(n)):
    if mat[i][0]>0.65:
        cal_mat.append(int(1))
    else:
        cal_mat.append(int(0))

correct=0

for i in range(int(n)):
    if inputy[i][0]==cal_mat[i]:
        correct+=1

print("Number of correct predictions = ",correct)
print("Percentage of correct predictions = ",((correct/n)*100))

mat=sess.run(y_val,feed_dict={x:inputx_r})

cal=[]

for i in range(int(50)):
    if mat[i][0]>0.6:
        cal.append(int(1))
    else:
        cal.append(int(0))

print(cal)
correct=0

for i in range(int(50)):
    if inputy_r[i]==cal[i]:
        correct+=1

print("Number of correct predictions = ",correct)
print("Percentage of correct predictions = ",((correct/(50))*100))


saver=tf.train.Saver()
saver.save(sess, "/home/arindam/Desktop/tfmodel/tfmodel")

