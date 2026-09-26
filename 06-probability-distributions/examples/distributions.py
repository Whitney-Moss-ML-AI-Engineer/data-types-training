"""Plot common probability distributions."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(-4, 4, 400)
normal_pdf = stats.norm.pdf(x, loc=0, scale=1)

plt.plot(x, normal_pdf, label="Normal PDF")
plt.title("Standard Normal Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()

k = np.arange(0, 16)
poisson_pmf = stats.poisson.pmf(k, mu=5)

plt.stem(k, poisson_pmf)
plt.title("Poisson Distribution (lambda=5)")
plt.xlabel("k")
plt.ylabel("P(X=k)")
plt.show()
