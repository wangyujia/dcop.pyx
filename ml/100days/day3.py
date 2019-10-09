# -*- coding: GB2312 -*-



###########################################################
#Day 3: 多元线性回归



# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('./py/ml/100days/datasets/50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Dummy Variable Trap
X = X[: , 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# regressor.fit(X_train, Y_train)
regressor = regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print('X_train', X_train)
print('Y_train', Y_train)
print('X_test', X_test)
print('Y_test', Y_test)
print('y_pred', y_pred)

# regression evaluation
from sklearn.metrics import r2_score
print(r2_score(Y_test, y_pred))



# I add
print("___train___", X_train[ : , 2])
print("___test___", X_test[ : , 2])

# Visualizing the test results
plt.scatter(X_train[ : , 2], Y_train, color = 'red')
plt.plot(X_train[ : , 2], regressor.predict(X_train), color = 'blue')
# plt.scatter(X_test[ : , 2], Y_test, color = 'green')
# plt.plot(X_test[ : , 2], regressor.predict(X_test), color = 'orange')
plt.show()

