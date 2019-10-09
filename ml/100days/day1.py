# -*- coding: GB2312 -*-



###########################################################
#Day 1: 数据预处理



#Step 1: 导入需要的库。numpy包含数学计算函数，pandas用于管理和导入数据集。我们一般使用 pandas 处理分析数据
import numpy as np
import pandas as pd


#Step 2: 导入数据集
dataset = pd.read_csv("./py/ml/100days/datasets/Data.csv")
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
print("Step 2: Importing dataset")
print("X")
print(X)
print("Y")
print(Y)

#Step 3: 处理丢失数据 (使用sklearn.preprocessing库的Imputer类) (可以用平均值或者中间值替换丢失的数据)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
print("---------------------")
print("Step 3: Handling the missing data")
print("step2")
print("X")
print(X)

#Step 4: 解析分类数据 (使用sklearn.preprocessing库的LabelEncoder类) (把"Yes"/"No"这样的字符串转换为数字)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
#Creating a dummy variable
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print("---------------------")
print("Step 4: Encoding categorical data")
print("X")
print(X)
print("Y")
print(Y)

#Step 5: 拆分数据集为训练集合和测试集合 (一个用来训练模型，一个用来验证模型) (二者比例为80:20)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
print("---------------------")
print("Step 5: Splitting the datasets into training sets and Test sets")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)

#Step 6: 特征缩放 (大部分模型算法使用两点间的'欧式距离'表示，但是特征在幅度、单位和范围姿势问题上变化很大，
# 在距离计算中，高幅度的特征比低幅度特征的权重更大。可用特征标准化或者Z值归一化解决)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("---------------------")
print("Step 6: Feature Scaling")
print("X_train")
print(X_train)
print("X_test")
print(X_test)