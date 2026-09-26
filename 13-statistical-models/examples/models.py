"""Introductory statistical and predictive models."""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 8, 10])

linear = LinearRegression().fit(X, y)
tree = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, y)

print("Linear prediction for 6:", linear.predict([[6]])[0])
print("Tree prediction for 6:", tree.predict([[6]])[0])

X_class = np.array([[1], [2], [3], [4], [5], [6]])
y_class = np.array([0, 0, 0, 1, 1, 1])

logistic = LogisticRegression().fit(X_class, y_class)
knn = KNeighborsClassifier(n_neighbors=3).fit(X_class, y_class)

print("Logistic class for 3.5:", logistic.predict([[3.5]])[0])
print("KNN class for 3.5:", knn.predict([[3.5]])[0])
