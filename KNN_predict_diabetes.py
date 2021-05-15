import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

dataset = pd.read_csv('diabetes.csv')
print(len(dataset))
dataset.head()

print(dataset.isnull().sum())

#replace zeroes
zeroes_not_accepted = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']

for column in zeroes_not_accepted:
    dataset[column] = dataset[column].replace(0,np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column]=dataset[column].replace(np.NaN, mean)

#split dataset
X = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0, test_size=0.2)

#feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#how to determine what value of k neighbor to use
import math
math.sqrt(len(y_test))

#if it is an even number integer, then try to make it odd. we can determine a vote with even number. Hence we make it odd and choose 11
#define the model : Init K-NN
classifier = KNeighborsClassifier(n_neighbors =11, p=2, metric ='euclidean')
#p represents if it's diabetic or not

#fit the model
classifier.fit(X_train, y_train)

#predict the test set results
y_pred = classifier.predict(X_test)
y_pred


#Evaluate the model using confusion matrix
#confusion matrix is a performance measurement for machine learning classification
cm = confusion_matrix(y_test, y_pred)
print(cm)

#check if false positive and false negative are crucial
f1_score(y_test,y_pred)

#check if true positive and true negative are crucial
accuracy_score(y_test, y_pred)

