"""Discrete and continuous random-variable examples."""
from scipy import stats

bernoulli = stats.bernoulli(p=0.70)
print("Bernoulli P(X=1):", bernoulli.pmf(1))
print("Bernoulli E[X]:", bernoulli.mean())
print("Bernoulli Var(X):", bernoulli.var())

normal = stats.norm(loc=100, scale=15)
print("Normal PDF at 100:", normal.pdf(100))
print("Normal CDF at 120:", normal.cdf(120))
print("P(X <= 120):", normal.cdf(120))
