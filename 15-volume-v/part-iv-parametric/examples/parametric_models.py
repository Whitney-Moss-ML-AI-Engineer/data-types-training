"""Part IV: parametric statistical and ML models."""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_regression, make_classification
import statsmodels.api as sm

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 3))
y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(size=100)

# 46 Linear Regression
print("Linear Regression:", LinearRegression().fit(X[:, [0]], y).coef_)

# 47 Multiple Linear Regression
print("Multiple Linear Regression:", LinearRegression().fit(X, y).coef_)

# 48 Polynomial Regression
poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X[:, [0]], y)
print("Polynomial prediction:", poly.predict([[0.5]])[0])

# 49 Logistic Regression
Xc, yc = make_classification(n_samples=100, n_features=3, random_state=42)
print("Logistic prediction:", LogisticRegression(max_iter=1000).fit(Xc, yc).predict([Xc[0]])[0])

# 50 Probit Regression
X_sm = sm.add_constant(Xc)
probit = sm.Probit(yc, X_sm).fit(disp=False)
print("Probit probability:", probit.predict(X_sm[:1])[0])

# 51 Poisson Regression
counts = rng.poisson(np.exp(.5 + .4 * X[:, 0]))
poisson = sm.GLM(counts, sm.add_constant(X[:, [0]]), family=sm.families.Poisson()).fit()
print("Poisson prediction:", poisson.predict([1, .5])[0])

# 52 Negative Binomial Regression
nb = sm.GLM(counts, sm.add_constant(X[:, [0]]), family=sm.families.NegativeBinomial()).fit()
print("Negative Binomial prediction:", nb.predict([1, .5])[0])

# 53 Ridge
print("Ridge:", Ridge(alpha=1).fit(X, y).coef_)

# 54 Lasso
print("Lasso:", Lasso(alpha=.05).fit(X, y).coef_)

# 55 Elastic Net
print("Elastic Net:", ElasticNet(alpha=.05, l1_ratio=.5).fit(X, y).coef_)

# 56 Bayesian Linear Regression (conjugate normal prior via closed-form posterior mean)
Xb = np.column_stack([np.ones(len(X)), X])
prior_precision = np.eye(Xb.shape[1])
posterior_mean = np.linalg.solve(Xb.T @ Xb + prior_precision, Xb.T @ y)
print("Bayesian linear posterior mean:", posterior_mean)

# 57 GLM
glm = sm.GLM(y, sm.add_constant(X), family=sm.families.Gaussian()).fit()
print("GLM prediction:", glm.predict(sm.add_constant(X[:1]))[0])

# 58 GAM-style parametric component using polynomial basis
print("GAM-style polynomial basis prediction:", poly.predict([[.2]])[0])

# 59 Linear Mixed Effects
groups = np.repeat(np.arange(10), 10)
mixed = sm.MixedLM(y, sm.add_constant(X[:, [0]]), groups=groups).fit()
print("Mixed-effects fixed effects:", mixed.fe_params)

# 60 Nonlinear Regression
from scipy.optimize import curve_fit
def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c
x = np.linspace(0, 2, 100)
yn = exponential_model(x, 2, .5, 1) + rng.normal(0, .2, 100)
params, _ = curve_fit(exponential_model, x, yn, p0=[1, .1, 0])
print("Nonlinear parameters:", params)

# 61 Gaussian Distribution Model
mu, sigma = np.mean(y), np.std(y, ddof=1)
print("Gaussian model:", {"mean": mu, "std": sigma})

# 62 Exponential Family: Poisson is a member.
print("Exponential-family log-likelihood example:", poisson.llf)

# 63 Hidden Markov Model: simple two-state simulation (requires hmmlearn for fitted HMM).
states = rng.choice([0, 1], size=20)
print("Hidden states example:", states)

# 64 Gaussian Naive Bayes
gnb = GaussianNB().fit(Xc, yc)
print("Gaussian NB:", gnb.predict(Xc[:1])[0])

# 65 Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis().fit(Xc, yc)
print("LDA:", lda.predict(Xc[:1])[0])
