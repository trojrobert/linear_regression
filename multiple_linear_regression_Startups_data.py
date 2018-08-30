
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


#import dataset 
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4].values

#Encoding the categorial variable 
#Encoding the Indepedence variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labEncoder_X = LabelEncoder()
X[: ,3] = labEncoder_X.fit_transform(X[: ,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X [:, 1:]

#Split the data 
from sklearn.model_selection import train_test_split 
train_X, test_X, train_y , test_y = train_test_split(X, y ,test_size = 0.2, random_state = 0)


#Create the model 
from sklearn.linear_model import LinearRegression 
linreg = LinearRegression()

#Fit and train the model
linreg.fit(train_X,train_y)

#Predict 
prediction = linreg.predict(test_X)

#Building the optimal model using Backward elimination 
import statsmodels.formula.api as sm

X =  np.append(arr =np.ones((50,1)),values = X,axis =1)
X_opt = X [:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()