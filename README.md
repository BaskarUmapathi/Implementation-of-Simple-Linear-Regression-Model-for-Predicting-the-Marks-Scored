# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1.Import the standard Libraries. 
Step 2.Set variables for assigning dataset values. 
Step 3.Import linear regression from sklearn. 
Step 4.Assign the points for representing in the graph. 
Step 5.Predict the regression for marks by using the representation of the graph. 
Step 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Developed by: BASKAR.U
RegisterNumber:  212223220013
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
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
```

## Output:
### df.head()
![image](https://github.com/user-attachments/assets/d8907bd6-c92b-4955-85d7-e7e7c6f3ebe9)
### df.tail()
![image](https://github.com/user-attachments/assets/99c7b98c-9de9-49a0-8f90-25b1248678df)
### Array value of X
![image](https://github.com/user-attachments/assets/29144f0e-4317-4ef9-9eb1-a3810d82eba4)
### Array value of Y
![image](https://github.com/user-attachments/assets/724294d5-9649-468d-86c8-70b7f015e814)
### Values of Y prediction
![image](https://github.com/user-attachments/assets/7d9796c1-ce05-4784-bace-0b09d6d37292)
### Array values of Y test
![image](https://github.com/user-attachments/assets/74a16419-92a4-4d2c-a3cd-d7c040b97a27)
### Training Set Graph
![image](https://github.com/user-attachments/assets/503e341c-7bcc-4805-b87e-c7132be99ea1)
### Test Set Graph
![image](https://github.com/user-attachments/assets/0afd4a15-b656-4af8-b312-db7c1aaf973c)
### Values of MSE, MAE and RMSE
![image](https://github.com/user-attachments/assets/9522b7c6-48f9-4b44-8972-7424d197db14)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
