import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/amartya/Desktop/Udemy_data/PCA/PCA/Wine.csv')

X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values

'''From the m independent variables, PCA extracts p<=m  new independent variables
that separates the most of the classes of the dependent variable.
Thus,LDA is an supervised model.
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)

'''Feature Scaling'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''Applying LDA'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#pca = PCA(n_components = None)
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

'''Logistic Regression'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

'''Predicting the test set result'''
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''3 Classes thus confusion matrrix of 3 dimensions.'''

'''Visualizing the results on the training set.'''

'''This code marks the decision boundary for classification.Decision boundary is straight line bcz logistic regression 
classifier is linear.'''
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
            alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c = ListedColormap(('red','green','blue'))(i), label = j)
plt.title('Logistic Regression(Training Set)')
plt.xlabel('LD1')
plt.ylabel('LD2') 
plt.legend()   
plt.show()

'''Visualizing the test set results.'''

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
            alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c = ListedColormap(('red','green','blue'))(i), label = j)
plt.title('Logistic Regression(Test Set)')
plt.xlabel('LD1')
plt.ylabel('LD2') 
plt.legend()   
plt.show()


