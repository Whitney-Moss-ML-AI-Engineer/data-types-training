"""Basic time-series operations."""
import pandas as pd
import matplotlib.pyplot as plt

dates = pd.date_range("2026-01-01", periods=12, freq="MS")
sales = pd.Series([100, 108, 105, 115, 122, 119, 130, 128, 140, 145, 151, 160], index=dates)

df = pd.DataFrame({"sales": sales})
df["lag_1"] = df["sales"].shift(1)
df["change"] = df["sales"].diff()
df["rolling_3"] = df["sales"].rolling(3).mean()

print(df)

df["sales"].plot(title="Monthly Sales")
plt.ylabel("Sales")
plt.show()
