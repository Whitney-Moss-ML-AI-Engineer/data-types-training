"""Convert text into numeric features with TF-IDF."""
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "machine learning models analyze data",
    "data science uses statistical models",
    "machine learning supports predictive analytics",
]

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(documents)

print("Features:", vectorizer.get_feature_names_out())
print("TF-IDF matrix shape:", matrix.shape)
print(matrix.toarray())
