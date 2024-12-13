# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries for data handling, machine learning, and evaluation. Fetch the California housing dataset and create a DataFrame with features and the target.
2. Select the first three features and combine the target with the seventh feature.Split the data into training and testing sets.
3. Apply StandardScaler to normalize both X and Y for training and testing datasets
4. Initialize an SGDRegressor model.Use MultiOutputRegressor to handle multiple outputs.
5. Fit the model to the scaled training data.
6. Predict on the test set.Inverse transform the predictions and test data to their original scale
7. Calculate the Mean Squared Error (MSE) between the predicted and actual values.Print the MSE and display the first few predictions.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: VESLIN ANISH A
RegisterNumber:212223240175
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


data=fetch_california_housing()
x=data.data[:,:3]
y=np.column_stack((data.target,data.data[:, 6]))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)

sgd=SGDRegressor(max_iter =1000,tol= 1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",y_pred[:5])
*/
```

## Output:

![Screenshot 2024-10-18 110349](https://github.com/user-attachments/assets/6f248bd3-9b16-470f-8c3d-b13b0ee2632a)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
