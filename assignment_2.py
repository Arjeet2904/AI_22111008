import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_california_housing, load_iris, load_breast_cancer, load_digits, fetch_openml

# Function to split data and scale features
def prepare_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

# 1. Linear Regression
print("1. Linear Regression")
california = fetch_california_housing()
X, y = california.data, california.target
X_train, X_test, y_train, y_test = prepare_data(X, y)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_lr)
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression: Actual vs Predicted House Prices")
plt.show()

# 2. K-Nearest Neighbors
print("\n2. K-Nearest Neighbors")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = prepare_data(X, y)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy: {accuracy_knn}")

# 3. Decision Trees
print("\n3. Decision Trees")
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = prepare_data(X, y)

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy: {accuracy_dt}")

# 4. Support Vector Machines
print("\n4. Support Vector Machines")
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = prepare_data(X, y)

model_svm = SVC(kernel='rbf', random_state=42)
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy: {accuracy_svm}")

# 5. Neural Networks
print("\n5. Neural Networks")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data / 255., mnist.target
X_train, X_test, y_train, y_test = prepare_data(X, y)

model_nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, random_state=42)
model_nn.fit(X_train, y_train)
y_pred_nn = model_nn.predict(X_test)
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print(f"Accuracy: {accuracy_nn}")