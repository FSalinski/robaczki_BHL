import time
import numpy as np
from pathlib import Path

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from codecarbon import EmissionsTracker


def load_datasets():
    d = {}

    # 1. Titanic – przewidywanie przeżycia
    titanic = sns.load_dataset("titanic").dropna(
        subset=["survived", "pclass", "age", "sibsp", "parch", "fare"]
    )
    X_tit = titanic[["pclass", "age", "sibsp", "parch", "fare"]].to_numpy()
    y_tit = titanic["survived"].to_numpy()
    d["titanic"] = (X_tit, y_tit)

    # 2. Penguins – Adelie vs reszta
    peng = sns.load_dataset("penguins").dropna(
        subset=["species", "bill_length_mm", "bill_depth_mm",
                "flipper_length_mm", "body_mass_g"]
    )
    X_peng = peng[
        ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    ].to_numpy()
    y_peng = (peng["species"] == "Adelie").astype(int).to_numpy()
    d["penguins_adelie"] = (X_peng, y_peng)

    # 3. Tips – wysoki vs niski napiwek
    tips = sns.load_dataset("tips").dropna(subset=["total_bill", "tip", "size"])
    med_tip = tips["tip"].median()
    y_tips = (tips["tip"] > med_tip).astype(int).to_numpy()
    X_tips = tips[["total_bill", "size"]].to_numpy()
    d["tips_high_tip"] = (X_tips, y_tips)

    # 4. Diamonds – drogi vs tani diament
    diamonds = sns.load_dataset("diamonds").dropna()
    if len(diamonds) > 8000:  # żeby nie zabić laptopa
        diamonds = diamonds.sample(n=8000, random_state=0)
    med_price = diamonds["price"].median()
    y_dia = (diamonds["price"] > med_price).astype(int).to_numpy()
    X_dia = diamonds[["carat", "depth", "table", "x", "y", "z"]].to_numpy()
    d["diamonds_expensive"] = (X_dia, y_dia)

    return d


def sample_random_config(family: str, rng: np.random.Generator):
    if family == "log_reg":
        C = 10 ** rng.uniform(-2, 2)
        max_iter = int(rng.integers(200, 801))
        params = {
            "C": C,
            "max_iter": max_iter,
            "solver": "lbfgs",
            "multi_class": "auto",
        }

    elif family == "random_forest":
        n_estimators = int(rng.integers(20, 501))
        max_depth = int(rng.integers(3, 21))
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "n_jobs": -1,
        }

    else:  # mlp
        n_layers = int(rng.integers(1, 4))
        hidden = tuple(int(rng.integers(10, 201)) for _ in range(n_layers))
        max_iter = int(rng.integers(50, 201))
        params = {
            "hidden_layer_sizes": hidden,
            "max_iter": max_iter,
            "activation": "relu",
            "solver": "adam",
        }

    return params


def build_model(family, params, seed: int):
    if family == "log_reg":
        clf = LogisticRegression(random_state=seed, **params)
    elif family == "random_forest":
        clf = RandomForestClassifier(random_state=seed, **params)
    else:
        clf = MLPClassifier(random_state=seed, **params)

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def binarize(y):
    cls0 = np.unique(y)[0]
    return (y == cls0).astype(int)


def run_experiment(dataset_name, X, y, family, params, exp_id, seed,
                   data_fraction, output_dir: Path, logger):
    y = binarize(y)

    data_fraction = float(data_fraction)
    subset_size = max(50, int(len(y) * data_fraction))
    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=subset_size,
        stratify=y,
        random_state=seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub,
        y_sub,
        test_size=0.2,
        random_state=seed,
        stratify=y_sub,
    )

    model = build_model(family, params, seed=seed)
    tracker = EmissionsTracker(
        project_name="hackathon_energy_ml",
        output_dir=str(output_dir),
        log_level="warning",
        save_to_file=True,
    )

    tracker.start()
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    emissions_kg = tracker.stop()

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    logger.info(
        f"[{exp_id}] {dataset_name} {family} seed={seed} "
        f"frac={data_fraction:.2f} time={t1-t0:.2f}s CO2={emissions_kg:.6f} AUC={auc:.3f}"
    )

    return {
        "experiment_id": exp_id,
        "dataset_name": dataset_name,
        "seed": seed,
        "data_fraction": data_fraction,
        "n_samples": X_sub.shape[0],
        "n_features": X_sub.shape[1],
        "model_family": family,
        "params": str(params),
        "train_time_s": t1 - t0,
        "roc_auc": auc,
        "emissions_kg": emissions_kg,
    }
