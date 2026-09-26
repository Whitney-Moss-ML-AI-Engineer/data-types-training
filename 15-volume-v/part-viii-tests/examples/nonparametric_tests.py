"""Part VIII: common nonparametric statistical tests."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
a = rng.normal(0, 1, 30)
b = rng.normal(.5, 1, 30)
paired_a = rng.normal(0,1,20)
paired_b = paired_a + rng.normal(.2,.5,20)
groups = [rng.normal(i,1,20) for i in range(3)]

# 101 Mann-Whitney U
print("101 Mann-Whitney:", stats.mannwhitneyu(a,b))

# 102 Wilcoxon Signed-Rank
print("102 Wilcoxon:", stats.wilcoxon(paired_a, paired_b))

# 103 Kruskal-Wallis
print("103 Kruskal-Wallis:", stats.kruskal(*groups))

# 104 Friedman
repeated = np.column_stack([rng.normal(i,1,20) for i in range(3)])
print("104 Friedman:", stats.friedmanchisquare(*repeated.T))

# 105 Kolmogorov-Smirnov
print("105 KS:", stats.kstest(a, "norm"))

# 106 Anderson-Darling
print("106 Anderson-Darling:", stats.anderson(a, dist="norm"))

# 107 Spearman Rank Correlation
print("107 Spearman:", stats.spearmanr(a,b))

# 108 Kendall Tau
print("108 Kendall:", stats.kendalltau(a,b))

# 109 Sign Test: binomial test on positive paired differences.
differences = paired_b - paired_a
positive = np.sum(differences > 0)
print("109 Sign test:", stats.binomtest(positive, len(differences), .5))

# 110 Runs Test: Wald-Wolfowitz implementation.
def runs_test(values):
    signs = np.where(np.asarray(values) >= np.median(values), 1, 0)
    runs = 1 + np.sum(signs[1:] != signs[:-1])
    n1, n0 = signs.sum(), len(signs)-signs.sum()
    expected = 1 + 2*n1*n0/(n1+n0)
    variance = (2*n1*n0*(2*n1*n0-n1-n0))/(((n1+n0)**2)*(n1+n0-1))
    z = (runs-expected)/np.sqrt(variance)
    return runs, z

print("110 Runs test:", runs_test(a))
