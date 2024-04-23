# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Vishal S
RegisterNumber:  212223110063
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize

# Load the data using pandas
data = pd.read_csv("/content/Mall_Customers.csv")

# Extract features (Age and Annual Income) and labels (Gender)
X = data[['Age', 'Annual Income (k$)']].values
y = data['Gender'].apply(lambda x: 1 if x == 'Female' else 0).values

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define cost function and gradient
def costFunction(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J, grad

# Define prediction function
def predict(theta, X):
    prob = sigmoid(np.dot(X, theta))
    return (prob >= 0.5).astype(int)

# Add intercept term to X
X_train = np.hstack((np.ones((X.shape[0], 1)), X))

# Initialize parameters
theta = np.zeros(X_train.shape[1])

# Calculate cost and gradient for initial parameters
J, grad = costFunction(theta, X_train, y)
print("Initial Cost:", J)
print("Initial Gradient:", grad)

# Optimize parameters using minimize function
res = optimize.minimize(fun=costFunction, x0=theta, args=(X_train, y), method='Newton-CG', jac=True)
optimal_theta = res.x

# Predict and calculate accuracy
accuracy = np.mean(predict(optimal_theta, X_train) == y)
print("Accuracy:", accuracy)

# Plot decision boundary
def plotDecisionBoundary(theta, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    X_plot = np.hstack((np.ones((X_plot.shape[0], 1)), X_plot))
    y_plot = np.dot(X_plot, theta).reshape(xx.shape)

    plt.figure()
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Female")
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Male")
    plt.contour(xx, yy, y_plot, levels=[0.5])
    plt.xlabel("Age")
    plt.ylabel("Annual Income (k$)")
    plt.legend()
    plt.show()

plotDecisionBoundary(optimal_theta, X, y)
def predict(theta,x):
  x_train = np.hstack((np.ones((x.shape[0],1)),x))
  prob = sigmoid(np.dot(x_train,theta))
  return (prob >=0.5).astype(int)


print('Prediction value of mean:')
np.mean(predict(res.x,X)==y)

```

## Output:

![WhatsApp Image 2024-04-23 at 09 07 05_2dd9de7b](https://github.com/vishal23000591/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147139719/ed6c6c91-0c2a-4844-af70-eb2baf50d30c)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

