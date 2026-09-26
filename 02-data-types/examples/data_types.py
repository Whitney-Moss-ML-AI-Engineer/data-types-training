"""Examples of common data types in Python and pandas."""

import pandas as pd

# Qualitative / categorical data
department = "Finance"
risk_level = "High"

# Quantitative data
age = 39
salary = 85000.50

# Discrete quantitative data: countable values
number_of_claims = 12

# Continuous quantitative data: measurable values
temperature = 72.6

# Build a small dataset
df = pd.DataFrame({
    "department": ["Finance", "IT", "HR"],
    "risk_level": ["High", "Low", "Medium"],
    "age": [39, 42, 35],
    "number_of_claims": [12, 4, 7],
    "temperature": [72.6, 68.2, 70.4],
})

print(df)
print("\nPandas data types:")
print(df.dtypes)

# Convert categorical columns to pandas category dtype
df["department"] = df["department"].astype("category")
df["risk_level"] = df["risk_level"].astype("category")

print("\nAfter categorical conversion:")
print(df.dtypes)
