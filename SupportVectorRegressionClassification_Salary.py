#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
#Importing the Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

#X = X.reshape(-1, 1)
#y = y.reshape(-1, 1)

#print(X)
#print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#X_train =X_train.reshape(-1, 1)
#X_test =X_test.reshape(-1, 1


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.fit_transform(X_test)

#Fitting SVM to the Training Set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state=0)
classifier.fit(X_train, y_train)
#Predicting  the test set results
y_pred  = classifier.predict(X_test)

#Save the Model to the Disk
filename = 'SVR_Classification.pkl'
pkl_file = open(filename, 'wb')
model = pickle.dump(classifier, pkl_file)

#Load model from the Disk
pkl_file = open(filename,'rb')
pkl_model = pickle.load(pkl_file)

print (pkl_model)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
                                                  
# Visualising the Training set Results
from matplotlib.colors import ListedColormap
x_set, y_set = X_train, y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max()+1, step = 0.01),
                    np.arange(start = x_set[:, 1].min()-1,stop = x_set[:, 1].max()+1, step = 0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('SVM (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




