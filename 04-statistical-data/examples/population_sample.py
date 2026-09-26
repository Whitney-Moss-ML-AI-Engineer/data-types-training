"""Population vs. sample statistics."""
import numpy as np

population = np.array([10, 12, 15, 18, 20, 22, 25, 28, 30, 35])
population_mean = population.mean()
population_variance = population.var()

rng = np.random.default_rng(42)
sample = rng.choice(population, size=5, replace=False)
sample_mean = sample.mean()
sample_variance = sample.var(ddof=1)  # unbiased sample variance

print("Population:", population)
print("Population mean:", population_mean)
print("Population variance:", population_variance)
print("Sample:", sample)
print("Sample mean:", sample_mean)
print("Unbiased sample variance:", sample_variance)
