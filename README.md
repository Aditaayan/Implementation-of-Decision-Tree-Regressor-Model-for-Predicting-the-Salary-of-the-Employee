# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
STEP 1: Start
STEP 2: Load the salary dataset into a Pandas DataFrame and inspect the first few rows using data.head().
STEP 3: Check the dataset for missing values using data.isnull().sum() and inspect the data structure using data.info().
STEP 4: Preprocess the categorical data. Use LabelEncoder to convert the "Position" column into numerical values.
STEP 5: Define the feature matrix (X) by selecting the relevant columns (e.g., Position, Level), and set the target variable (Y) as the "Salary" column.
STEP 6: Split the dataset into training and testing sets using train_test_split() with a test size of 20%.
STEP 7: Initialize the Decision Tree Regressor and fit the model to the training data (x_train, y_train).
STEP 8: Predict the target values on the testing set (x_test) using dt.predict().
STEP 9: Calculate the Mean Squared Error (MSE) using metrics.mean_squared_error() and the R-squared score (r2_score()) to evaluate the model's performance.
STEP 10: Use the trained model to predict the salary of an employee with specific input features (dt.predict([[5,6]])).
STEP 11: End
``` 

## Program:
```py
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ADITAAYAN M
RegisterNumber: 212223040006
*/
import pandas as pd
data=pd.read_csv("C:/Users/SEC/Downloads/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
```
data.head()
```
![371068147-32b8c60f-3a05-421f-8bec-efa60eb22ba4](https://github.com/user-attachments/assets/cc1e0462-a638-4b2f-8d8f-77acd779905a)

```
mean_squared_error
```
![371068280-0c3b012c-8810-413c-88f1-8751a3373191](https://github.com/user-attachments/assets/708e8570-b791-4292-8381-966646112370)

```
r2:
```
![371068365-e1ce2a47-1466-4f43-8570-b7940c5058ad](https://github.com/user-attachments/assets/3fe6c1ac-248a-4b24-94d0-6a09db381ab1)

```
predicted
```
![371068485-de2a4ba8-65ea-4397-89f6-84fb8340fbd9](https://github.com/user-attachments/assets/0df6f4e9-b8e3-4645-bfeb-35d6ea0f79d3)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
