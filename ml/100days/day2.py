# -*- coding: GB2312 -*-



###########################################################
#Day 2: 简单线性回归

# 这是一种根据自变量X来预测因变量Y的方法，并且假设这两个变量是线性相关的。
#   y = b0 + b1x
#   Score = b0 + b1 * hours



# Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('./py/ml/100days/datasets/studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/4, random_state = 0)


# Fitting Simple Linear Regression Model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)


# Predecting the Result
Y_pred = regressor.predict(X_test)
print('X_train', X_train)
print('Y_train', Y_train)
print('X_test', X_test)
print('Y_test', Y_test)
print('Y_pred', Y_pred)


# Visualizing the test results
# plt.scatter(X_test, Y_test, color = 'red')
# plt.plot(X_test, regressor.predict(X_test), color = 'blue')
# plt.show()
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.scatter(X_test, Y_test, color = 'green')
plt.plot(X_test, regressor.predict(X_test), color = 'orange')
plt.show()
