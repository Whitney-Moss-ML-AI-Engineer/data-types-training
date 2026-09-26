"""Part II: 15 discrete probability distribution examples."""
import numpy as np
from scipy import stats

examples = {
    "Bernoulli": stats.bernoulli(p=0.6),
    "Binomial": stats.binom(n=10, p=0.6),
    "Negative Binomial": stats.nbinom(n=5, p=0.6),
    "Geometric": stats.geom(p=0.6),
    "Hypergeometric": stats.hypergeom(M=50, n=10, N=5),
    "Poisson": stats.poisson(mu=4),
    "Multinomial": stats.multinomial(n=10, p=[0.2, 0.3, 0.5]),
    "Discrete Uniform": stats.randint(low=1, high=7),
    "Zipf": stats.zipf(a=2),
}

for name, dist in examples.items():
    print(f"{name:20s} mean={dist.mean()} variance={dist.var()}")

# 30 Negative Hypergeometric trials: direct simulation.
rng = np.random.default_rng(42)
population = np.array([1] * 10 + [0] * 20)
draws = rng.permutation(population)
failures_before_3_successes = []
for _ in range(1000):
    draw = rng.permutation(population)
    successes = failures = 0
    for value in draw:
        if value:
            successes += 1
            if successes == 3:
                break
        else:
            failures += 1
    failures_before_3_successes.append(failures)
print("Negative Hypergeometric mean:", np.mean(failures_before_3_successes))

# Skellam: difference of two independent Poisson counts.
skellam = stats.skellam(mu1=8, mu2=5)
print("Skellam P(X=2):", skellam.pmf(2))

# Borel distribution: probability mass computed recursively from the definition.
def borel_pmf(k, lam=0.5):
    if k < 1:
        return 0.0
    return np.exp(-lam * k) * (lam * k) ** (k - 1) / np.math.factorial(k)

print("Borel P(X=3):", borel_pmf(3))

# Yule-Simon.
print("Yule-Simon P(X=3):", stats.yulesimon.pmf(3, a=2))

# Conway-Maxwell-Poisson: normalized finite approximation.
def cmp_pmf(k, lam=3, nu=1.5, max_k=100):
    weights = np.array([lam ** j / (np.math.factorial(j) ** nu) for j in range(max_k + 1)])
    return weights[k] / weights.sum()

print("COM-Poisson P(X=3):", cmp_pmf(3))

# Rademacher: X is -1 or +1 with equal probability.
print("Rademacher outcomes:", rng.choice([-1, 1], size=10))
