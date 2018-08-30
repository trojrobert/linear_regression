# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#create model 
from sklearn.linear_model import LinearRegression 
linearReg = LinearRegression()

#Fit and train the model
linearReg.fit(X_train,y_train)

#predicting test set 
predicition_test = linearReg.predict(X_test)
predicition_train = linearReg.predict(X_train)
#Visualizing train data 
plt.scatter(X_train,y_train, color ='red')
plt.plot(X_train,predicition_train, color ='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing test data 
plt.scatter(X_test,y_test, color ='red')
plt.plot(X_train,predicition_train, color ='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()