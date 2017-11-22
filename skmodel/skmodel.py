import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


df=pd.read_csv("/home/arindam/Desktop/tfmodel/data.csv")
in_data=df.drop(["Close","Volume"],axis=1)
label=df.drop(["Open","High","Low","Volume"],axis=1)

in_data.loc[:,("Result")]=in_data["Open"]<label["Close"]
in_data.loc[:,("Result")]=in_data["Result"].astype(int)

inputx=in_data.loc[:,["Open","High","Low"]].astype(np.float64).as_matrix()
print(inputx)
inputy=in_data.loc[:,("Result")].as_matrix()
print(inputy)

dfr=pd.read_csv("/home/arindam/Desktop/tfmodel/yrcw.csv")
in_data_r=dfr.drop(["Close","Volume"],axis=1)
label_r=dfr.drop(["Open","High","Low","Volume"],axis=1)
in_data_r.loc[:,("Result")]=in_data_r["Open"]<label_r["Close"]
in_data_r.loc[:,("Result")]=in_data_r["Result"].astype(int)

inputx_r=in_data_r.loc[:,["Open","High","Low"]].astype(np.float64).as_matrix()
print(inputx_r)
inputy_r=in_data_r.loc[:,("Result")].as_matrix()
print(inputy_r)


clf=svm.SVC(C=3, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)

clf.fit(inputx, inputy)

pred_res=clf.predict(inputx_r)

print(pred_res)

correct=0
n=len(inputy_r)
for i in range(n):
    if pred_res[i]==inputy_r[i]:
        correct+=1

print("Percentage correct(Linear) = ",((correct/n)*100))

clf=svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)

clf.fit(inputx, inputy)

pred_res=clf.predict(inputx_r)

print(pred_res)

correct=0
n=len(inputy_r)
for i in range(n):
    if pred_res[i]==inputy_r[i]:
        correct+=1

print("Percentage correct(RBF) = ",((correct/n)*100))

clf=svm.SVC(C=1, cache_size=200, class_weight=None, coef0=1.0,
            decision_function_shape=None, degree=2, gamma='auto', kernel='sigmoid',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)

clf.fit(inputx, inputy)

pred_res=clf.predict(inputx_r)

print(pred_res)

correct=0
n=len(inputy_r)
for i in range(n):
    if pred_res[i]==inputy_r[i]:
        correct+=1

print("Percentage correct(Sigmoid) = ",((correct/n)*100))

print("No. of training set entries = ",len(inputx))

