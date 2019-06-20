import sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#Setting Random seed
np.random.seed(0)

iris = load_iris()
df = pd.DataFrame(iris.data, columns= iris.feature_frame)
df.head()

print (iris)
# Adding a new column for the species name
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(df.head())

#Creating Train and Test Data
df['is_train'] = np.random.uniform(0, 1, len(df))  <= .75

#Creating dataframes with test rows and tarining rows
train, test = df[df['is_train'] == True], df[[ 'is_train'] ==False]
print('Numbers of observations in the training data:',  len(train))
print('Numbers of observations in the test data:', len(test))

features = df.columns[:4]
print(features)

#Converting each species name into digits
y = pd.factorize(train['species'])[0]

# Creating a RandomForestClassifier
clf = RandomForestClassifier(n_jobs = 2 , random_state=0)

# Training the classifier
clf.fit(train[features], y)
#Applying the trained Classifier to the test
clf.predict(test[features])
clf.predict_proba(test[features])[0:10]

# Mapping nmaes for the plants for each predicted plant class
preds = iris.target_names[clf.predict(test[features])]

print(preds[0:5])

#Creating confusion matrix
pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames =['Predicted Species'])




