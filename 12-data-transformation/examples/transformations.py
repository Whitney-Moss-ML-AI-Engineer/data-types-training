"""Scale numeric data and encode categorical data."""
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

df = pd.DataFrame({
    "income": [40000, 60000, 80000],
    "age": [25, 40, 55],
    "risk": ["low", "medium", "high"],
})

numeric = df[["income", "age"]]

standard = StandardScaler().fit_transform(numeric)
minimum_maximum = MinMaxScaler().fit_transform(numeric)

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
risk_encoded = encoder.fit_transform(df[["risk"]])

print("Standardized:\n", standard)
print("\nMin-max scaled:\n", minimum_maximum)
print("\nOne-hot categories:", encoder.categories_)
print(risk_encoded)
