"""Common data structures used in data science."""
import numpy as np
import pandas as pd

scalar = 7
vector = np.array([1, 2, 3])
matrix = np.array([[1, 2], [3, 4]])
tensor = np.arange(24).reshape(2, 3, 4)

df = pd.DataFrame({
    "customer_id": [101, 102, 103],
    "sales": [250.0, 180.0, 420.0]
})

print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix:\n", matrix)
print("Tensor shape:", tensor.shape)
print("DataFrame:\n", df)
