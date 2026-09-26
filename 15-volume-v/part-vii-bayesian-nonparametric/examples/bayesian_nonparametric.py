"""Part VII: Bayesian nonparametric examples."""
import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.mixture import BayesianGaussianMixture
from sklearn.ensemble import RandomForestRegressor

rng = np.random.default_rng(42)
X = np.linspace(0, 10, 80).reshape(-1,1)
y = np.sin(X[:,0]) + rng.normal(0,.1,80)

# 91 Gaussian Process
gp = GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(.1), random_state=42).fit(X, y)
print("91 GP prediction:", gp.predict([[5]])[0])

# 92 Dirichlet Process: stick-breaking weights.
weights = []
remaining = 1.0
for _ in range(10):
    v = rng.beta(1, 1)
    weights.append(remaining*v)
    remaining *= (1-v)
print("92 DP weights:", weights)

# 93 Chinese Restaurant Process.
def crp(n, alpha=1):
    tables = [1]
    for _ in range(1, n):
        probs = np.array(tables + [alpha], dtype=float)
        probs /= probs.sum()
        choice = rng.choice(len(probs), p=probs)
        if choice == len(tables):
            tables.append(1)
        else:
            tables[choice] += 1
    return tables
print("93 CRP table sizes:", crp(20))

# 94 Indian Buffet Process.
def ibp(n, alpha=1):
    dishes = []
    for i in range(n):
        if i == 0:
            dishes.append([1])
            continue
        existing = rng.binomial(1, np.array([sum(row[j] for row in dishes) for j in range(len(dishes[0]))])/(i+1))
        new = rng.poisson(alpha/(i+1))
        dishes.append(list(existing)+[1]*new)
    return dishes
print("94 IBP rows:", ibp(5))

# 95 Hierarchical DP: shared base-measure draws.
global_topics = rng.dirichlet([1,1,1])
group_topics = [rng.dirichlet(5*global_topics) for _ in range(3)]
print("95 HDP group topics:", group_topics)

# 96 Beta Process: random beta weights.
print("96 Beta process weights:", rng.beta(.5, 2, size=10))

# 97 Pitman-Yor Process: discount + concentration.
def pitman_yor(n, alpha=.5, discount=.2):
    tables = [1]
    for _ in range(1,n):
        masses = [size-discount for size in tables] + [alpha+discount*len(tables)]
        masses = np.array(masses)/sum(masses)
        choice = rng.choice(len(masses), p=masses)
        if choice == len(tables): tables.append(1)
        else: tables[choice] += 1
    return tables
print("97 Pitman-Yor:", pitman_yor(20))

# 98 Gaussian Process Regression
print("98 GP regression prediction:", gp.predict([[2]])[0])

# 99 Bayesian Additive Regression Trees (BART): sklearn ensemble as a practical tree-ensemble proxy.
bart_proxy = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
print("99 BART-style tree ensemble prediction:", bart_proxy.predict([[5]])[0])

# 100 Infinite Gaussian Mixture Model: Bayesian GMM with more components than expected clusters.
X2 = np.vstack([rng.normal(-2,.5,(100,1)), rng.normal(2,.5,(100,1))])
bgmm = BayesianGaussianMixture(n_components=10, weight_concentration_prior_type="dirichlet_process", random_state=42).fit(X2)
print("100 Infinite GMM active weights:", bgmm.weights_[bgmm.weights_ > .01])
