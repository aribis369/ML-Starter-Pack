import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values


import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrograms')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)

plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=50,c='red',label='cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=50,c='green',label='cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=50,c='cyan',label='cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=50,c='magenta',label='cluster 5')


plt.legend()
plt.show()