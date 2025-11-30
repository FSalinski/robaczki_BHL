import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Import funkcji eco_random_search
from eco_search import eco_random_search


# Wczytanie danych
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Obliczenie AUC modelu baseline
baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)
baseline_auc = roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1])

print("Baseline AUC =", baseline_auc)

# Rozkłady hiperparametrów do przeszukania
param_distributions = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 4, 8],
    "min_samples_leaf": [1, 2, 4],
}

# Uruchomienie eco_random_search
df_results, best_config = eco_random_search(
    X=X_train,
    y=y_train,
    param_distributions=param_distributions,
    data_fraction=0.10,
    baseline_auc=baseline_auc,
    n_iter=20,
    random_state=42,
    project_name="eco_demo"
)

print("\nNajlepsza konfiguracja wg EAUG:")
print(best_config)

print("\nTabela wyników:")
print(df_results.head())
