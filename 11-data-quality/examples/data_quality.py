"""Simple data-quality checks."""
import pandas as pd

df = pd.DataFrame({
    "customer_id": [1, 2, 2, 4],
    "age": [34, None, 220, 41],
    "sales": [1200, 850, 850, 1500],
})

print("Missing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nInvalid ages:", ((df["age"] < 0) | (df["age"] > 120)).sum())
print("\nSummary:\n", df.describe())
