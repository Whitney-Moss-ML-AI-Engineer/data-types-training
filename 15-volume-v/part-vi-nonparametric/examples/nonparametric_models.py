"""Part VI: nonparametric model examples."""
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 2))
y = X[:,0]**2 + rng.normal(0,.2,200)
yc = (X[:,0] > 0).astype(int)

# 76 KDE
kde = stats.gaussian_kde(X[:,0])
print("76 KDE density at 0:", kde.evaluate([0])[0])

# 77 KNN
knn = KNeighborsClassifier(n_neighbors=5).fit(X, yc)
print("77 KNN class:", knn.predict(X[:1])[0])

# 78 Decision Tree
tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)
print("78 Tree prediction:", tree.predict(X[:1])[0])

# 79 Random Forest
rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
print("79 Random Forest prediction:", rf.predict(X[:1])[0])

# 80 Gradient Boosted Trees
gb = GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X, y)
print("80 Gradient Boosting prediction:", gb.predict(X[:1])[0])

# 81 Kernel Regression: Nadaraya-Watson implementation.
def kernel_regression(x_train, y_train, x0, bandwidth=0.5):
    weights = np.exp(-((x_train-x0)**2)/(2*bandwidth**2))
    return np.sum(weights*y_train)/np.sum(weights)
print("81 Kernel regression:", kernel_regression(X[:,0], y, 0))

# 82 Local Polynomial Regression: local weighted linear fit.
def local_linear(x_train, y_train, x0, bandwidth=.5):
    w = np.exp(-((x_train-x0)**2)/(2*bandwidth**2))
    model = LinearRegression().fit((x_train-x0).reshape(-1,1), y_train, sample_weight=w)
    return model.predict([[0]])[0]
print("82 Local polynomial:", local_linear(X[:,0], y, 0))

# 83 LOESS / LOWESS
from statsmodels.nonparametric.smoothers_lowess import lowess
loess = lowess(y, X[:,0], frac=.3)
print("83 LOESS first fitted value:", loess[0])

# 84 Isotonic Regression
iso = IsotonicRegression().fit(np.sort(X[:,0]), np.sort(y))
print("84 Isotonic prediction:", iso.predict([0])[0])

# 85 Smoothing spline
from scipy.interpolate import UnivariateSpline
order = np.argsort(X[:,0])
smooth = UnivariateSpline(X[:,0][order], y[order], s=20)
print("85 Smoothing spline:", smooth(0))

# 86 Kernel smoothing (same concept, different bandwidth).
print("86 Kernel smoothing:", kernel_regression(X[:,0], y, 1, bandwidth=1.0))

# 87 Nearest-neighbor density estimation.
def nn_density(values, point, k=10):
    distances = np.sort(np.abs(values-point))
    radius = distances[k]
    return k / (len(values) * 2 * radius)
print("87 NN density:", nn_density(X[:,0], 0))

# 88 Empirical Distribution Function
sample = np.sort(X[:,0])
def edf(value):
    return np.mean(sample <= value)
print("88 EDF at 0:", edf(0))

# 89 Histogram Density Estimation
counts, edges = np.histogram(X[:,0], bins=10, density=True)
print("89 Histogram density:", counts)

# 90 Dirichlet Process: finite stick-breaking approximation.
weights, remaining = [], 1.0
for _ in range(5):
    beta = rng.beta(1, 1)
    weight = remaining * beta
    weights.append(weight)
    remaining -= weight
print("90 DP stick weights:", weights)
