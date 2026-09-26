"""Build a preprocessing + machine-learning pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

X = pd.DataFrame({
    "age": [22, 35, 47, 51, 29, 42],
    "income": [35000, 52000, 76000, 90000, 41000, 68000],
    "segment": ["A", "B", "B", "C", "A", "C"],
})
y = [0, 0, 1, 1, 0, 1]

numeric_features = ["age", "income"]
categorical_features = ["segment"]

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X, y)
print("Predictions:", model.predict(X))
print("Class probabilities:\n", model.predict_proba(X))
