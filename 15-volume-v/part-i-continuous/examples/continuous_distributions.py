"""Part I: 20 continuous distribution examples.
Run with: python continuous_distributions.py
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 400)
continuous = {
    "PDF": stats.norm,
    "CDF": stats.norm,
    "Normal": stats.norm,
    "Uniform": stats.uniform,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Beta": stats.beta,
    "Weibull": stats.weibull_min,
    "Logistic": stats.logistic,
    "Log-Normal": stats.lognorm,
    "Rayleigh": stats.rayleigh,
    "Pareto": stats.pareto,
    "Cauchy": stats.cauchy,
    "Laplace": stats.laplace,
    "Gumbel": stats.gumbel_r,
    "Student-t": stats.t,
    "Chi-Square": stats.chi2,
    "F": stats.f,
    "GEV": stats.genextreme,
    "Inverse Gaussian": stats.invgauss,
}

for name, distribution in continuous.items():
    if name == "Normal":
        value = distribution.pdf(0)
    elif name == "CDF":
        value = distribution.cdf(0)
    elif name == "Uniform":
        value = distribution.pdf(0.5)
    elif name in {"Beta"}:
        value = distribution.pdf(0.5, a=2, b=5)
    elif name in {"Gamma"}:
        value = distribution.pdf(2, a=3)
    elif name == "Weibull":
        value = distribution.pdf(1.5, c=2)
    elif name == "Log-Normal":
        value = distribution.pdf(1, s=0.5)
    elif name == "Student-t":
        value = distribution.pdf(0, df=5)
    elif name == "Chi-Square":
        value = distribution.pdf(2, df=5)
    elif name == "F":
        value = distribution.pdf(1, dfn=5, dfd=10)
    elif name == "GEV":
        value = distribution.pdf(0, c=0.1)
    elif name == "Inverse Gaussian":
        value = distribution.pdf(1, mu=1)
    else:
        value = distribution.pdf(0)
    print(f"{name:20s}: {value:.6f}")

# PDF and CDF are shown explicitly for the standard normal.
normal_pdf = stats.norm.pdf(x)
normal_cdf = stats.norm.cdf(x)

plt.plot(x, normal_pdf, label="PDF")
plt.plot(x, normal_cdf, label="CDF")
plt.title("Normal PDF and CDF")
plt.legend()
plt.show()
