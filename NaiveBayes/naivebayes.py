import pandas as pd
import numpy as np
import math
import pickle


                                                                               
#Creating a Dictionary with 2500 keys and setting their value to 1. The reason of putting the value of
#1 instead of zero is because of the laplace smoothing of the numerator.
#pickle module helps in seralization of data. It is easier to load data.

dic1={}          #dic1 contains words appeared in non spam emails.
dic2={}          #dic2 contains words appeared in spam emails.
for i in range(1,2501):
    dic1.update({i:1})
    dic2.update({i:1})
k=[dic1,dic2]
with open("dic.pickle","wb") as f:
    pickle.dump(k,f)

with open("dic.pickle","rb") as f:
    k=pickle.load(f)                                                    #k[0] contains words appeared in non spam emails.
                                                                                            #k[1] contains words appeared in spam emails.
v=2500
df=pd.read_csv("train-features.txt", sep=' ',
                  names = ["DocID", "DicID", "Occ"])
s=df["DocID"]

#reading the file and giving them respective headers
#DocId- Document number,DicID-Dictionary token number (1-2500),Occ-No. of times occured in the respective document.


##Training the classifier

c=1
r=0                       #Counting the length of each words in the document
a=[]                       #a is a list of all the lengths of document like a[0] is the no. of words in first document
for i in range(len(s)):
    if (s[i])==c:
        r+=df["Occ"][i]
    else:
        a.append(r)
        c+=1                                     
        r=r-r
        r+=df["Occ"][i] 
a.append(r)
b=a[0:350]             #Dividing the lenghts into two lists. As 0-350 documents are not spam(0) and 350-700 are spam(1) 
a=a[350:700]
nsp=sum(b)+v   #v is length of the dictionary ie 2500, it is added due to laplace smoothing
sp=sum(a)+v
sums=[nsp,sp]
with open("dicsum.pickle","wb") as f:
   pickle.dump(sums,f)

sums=[]
with open("dicsum.pickle","rb") as f:
   sums=pickle.load(f)
 




for i in range(len(s)):              #Updating the non spam and spam dictionary by adding the occurance of the word.
    if int(s[i])<=350:
        k[0][(df["DicID"][i])]+=df["Occ"][i]
    else:
        k[1][(df["DicID"][i])]+=df["Occ"][i]
            
with open("classydicl.pickle","wb") as f:
   pickle.dump(k,f)
   



with open("classydicl.pickle","rb") as f:
    q=pickle.load(f)                    #Our numerator and denominator are both ready.Now we Divide.

for keys in (q[0]):
    q[0][keys]=np.divide(q[0][keys],sums[0])
    q[1][keys]=np.divide(q[1][keys],sums[1])
    

with open("newclassydic.pickle","wb") as f:
   pickle.dump(q,f)


with open("newclassydic.pickle","rb") as f:
    k=pickle.load(f)

#newclassydic is our trained classifier
    



##Testing The Naive Bayes Classifier

df=pd.read_csv("test-features.txt", sep=' ',
                  names = ["DocID", "DicID", "Occ"])  #reading the file and giving them respective headers
s=df["DocID"]
t=df["DicID"]
u=df["Occ"]
x=np.log(0.50)                  #0.50 is the probability of spam and non spam dataset in our training data.
y=np.log(0.50)                  #x is the prob of non spam and y is the prob of of spam
                                                   #Applying the naive bayes algorithm.We are adding the log instead of multipying due to underflow.
z=1
arr=[]
for i in range(len(s)):
    if (s[i]==z):
        e=(k[0][t[i]])*(u[i])
        f=(k[1][t[i]])*(u[i])
        x+=np.log(e)
        y+=np.log(f)
    else:
        z+=1
        if x>y:
            arr.append(0)
        else:
            arr.append(1)
        x=np.log(0.50)
        y=np.log(0.50)
        e=(k[0][t[i]])*(u[i])
        f=(k[1][t[i]])*(u[i])
        x+=np.log(e)
        y+=np.log(f)
if x>y:
    arr.append(0)
else:
    arr.append(1)
df=pd.read_csv("test-labels.txt",names = ["LabelId"])  #reading the file and giving them respective header.
accuracy=0
l=df["LabelId"]
for i in range(len(arr)):      #Comparing test label and prediction(arr)
    if (l[i]==arr[i]):
        accuracy+=1
accuracy=accuracy/len(arr)
print ("Accuracy of the Naive Bayes Algorithm is",accuracy*100.0)
submission = pd.DataFrame(arr)
submission.to_csv('prediction.txt',index = False)#Creates prediction into a new file. 

    
