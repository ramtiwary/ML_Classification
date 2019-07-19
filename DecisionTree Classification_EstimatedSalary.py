#Importing the Libraries
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

#Importing The Data set
dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

print(x)
print(y)
#x = x.reshape(-1, 1)
#y = y.reshape(-1, 1)

# Handling categorical data
positions = pd.get_dummies(dataset['Gender'])
dataSet = dataset.drop('Gender', axis=1)
dataSet = pd.concat([dataSet, positions], axis=1)


#Spiliting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import  StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.tree import  DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion ='entropy',random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set Resilts
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max()+1, step = 0.01),
                    np.arange(start = x_set[:, 1].min()-1,stop = x_set[:, 1].max()+1, step = 0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Decision Tree (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Splitting dataset into 2 different csv files
df_training = dataSet.sample(frac=0.7)
df_test = pd.concat([dataSet, df_training]).drop_duplicates(keep=False)
length_new = len(dataSet.columns)
y_index = dataSet.columns.get_loc("EstimatedSalary")
df_training.to_csv('training_data.csv', header=True, index=None)
df_test.to_csv('test_data.csv', header=True, index=None)

#Save Model
file_name = 'DecisionTreeClassification.pkl'
pkl_file = open(file_name, 'wb')
model = pickle.dump(classifier, pkl_file)


#Load the model from the Disk



#Loading pickle model to predict data from test_data.csv file
pkl_file = open(file_name, 'rb')
model_pkl = pickle.load(pkl_file)

dataset_testdata = pd.read_csv('testdata.csv')

x_testdata = dataset_testdata.iloc[:, (len(data.columns)-1): len(dataSet)]
y_testdata = dataset_testdata.iloc[:, y_index:(y_index+1)]
y_pred_pkl = model_pkl.predict(np.array([6.5]).reshape(-1,1))





