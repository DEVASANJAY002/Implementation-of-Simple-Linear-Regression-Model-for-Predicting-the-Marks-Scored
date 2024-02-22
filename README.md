# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datase. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DEVASANJAY N
RegisterNumber:  212223040032
*/
```

## Output:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
Output:
Dataset
![image](https://github.com/DEVASANJAY002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152069249/e58ee326-830e-4a42-b9d7-95829da8d08d)

Head Values
output
![image](https://github.com/DEVASANJAY002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152069249/1132d907-627e-48e2-b667-0e8c2ff4da68)


Tail Values
output
![image](https://github.com/DEVASANJAY002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152069249/7dcc5fca-353b-4492-a856-b58e7421abc3)


X and Y values
output
![image](https://github.com/DEVASANJAY002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152069249/4b79919c-b0f8-44e8-a57e-1c8c8a69ce12)


Predication values of X and Y
output
![image](https://github.com/DEVASANJAY002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152069249/c6d11696-d44c-4f73-abdb-2279495e255a)

MSE,MAE and RMSE
![image](https://github.com/DEVASANJAY002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152069249/5c39f0a3-99e4-4b1f-a9cb-845151a1fc76)


Training Set
![image](https://github.com/DEVASANJAY002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152069249/726471e0-4cbc-4764-9aa8-2151100e26f2)


Testing Set
![image](https://github.com/DEVASANJAY002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152069249/1fd5bd8b-5d17-4e9d-b9ac-df37898420fc)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
