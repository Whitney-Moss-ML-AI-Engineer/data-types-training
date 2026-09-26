"""Examples of nominal, ordinal, interval, and ratio data."""

import pandas as pd

df = pd.DataFrame({
    # Nominal: categories without a natural order
    "department": ["Finance", "IT", "HR"],

    # Ordinal: categories with meaningful order
    "risk_level": ["High", "Low", "Medium"],

    # Interval: differences are meaningful; zero is not an absolute absence
    "temperature_c": [0, 20, 40],

    # Ratio: meaningful zero and meaningful ratios
    "revenue": [1000, 2000, 4000],
})

risk_order = pd.CategoricalDtype(
    categories=["Low", "Medium", "High"],
    ordered=True
)
df["risk_level"] = df["risk_level"].astype(risk_order)

print(df)
print("\nIs $4,000 twice $2,000?", 4000 / 2000 == 2)
print("Ordinal comparison:", df.loc[0, "risk_level"] > df.loc[1, "risk_level"])
