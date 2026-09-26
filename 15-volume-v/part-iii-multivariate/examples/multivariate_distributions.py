"""Part III: multivariate distribution examples."""
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

rng = np.random.default_rng(42)

# 36 Multivariate Normal
mean = [0, 0]
cov = [[1, 0.7], [0.7, 1]]
mvn = multivariate_normal(mean=mean, cov=cov)
print("Multivariate Normal PDF:", mvn.pdf([0, 0]))

# 37 Multinomial Logistic: softmax probabilities.
X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
W = np.array([[1.0, 0.5], [0.2, 1.0], [-0.5, 0.2]])
scores = X @ W.T
probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
probabilities /= probabilities.sum(axis=1, keepdims=True)
print("Multinomial logistic probabilities:\n", probabilities)

# 38 Dirichlet
alpha = np.array([2.0, 3.0, 5.0])
print("Dirichlet sample:", rng.dirichlet(alpha))

# 39 Wishart
wishart = stats.wishart(df=5, scale=np.eye(2))
print("Wishart sample:\n", wishart.rvs(random_state=rng))

# 40 Inverse Wishart
inverse_wishart = stats.invwishart(df=5, scale=np.eye(2))
print("Inverse Wishart sample:\n", inverse_wishart.rvs(random_state=rng))

# 41 Matrix Normal: sample via row/column covariance construction.
A = rng.normal(size=(2, 3))
row_cov = np.array([[1.0, .2], [.2, 1.0]])
col_cov = np.eye(3)
Lr, Lc = np.linalg.cholesky(row_cov), np.linalg.cholesky(col_cov)
matrix_normal_sample = Lr @ A @ Lc.T
print("Matrix Normal sample:\n", matrix_normal_sample)

# 42 Multivariate t: stochastic representation.
z = rng.multivariate_normal([0, 0], np.eye(2))
g = rng.chisquare(df=5)
print("Multivariate t sample:", z / np.sqrt(g / 5))

# 43 Dirichlet-Multinomial
p = rng.dirichlet([2, 3, 5])
print("Dirichlet-Multinomial counts:", rng.multinomial(20, p))

# 44 Gaussian Mixture
data = np.vstack([rng.normal(-2, .6, (100, 2)), rng.normal(2, .8, (100, 2))])
gmm = GaussianMixture(n_components=2, random_state=42).fit(data)
print("GMM labels:", gmm.predict(data[:5]))

# 45 Gaussian copula: correlated normals transformed to uniforms.
z = rng.multivariate_normal([0, 0], [[1, .8], [.8, 1]], size=5)
u = stats.norm.cdf(z)
print("Copula uniform samples:\n", u)
