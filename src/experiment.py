import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from codecarbon import EmissionsTracker

from eco_search import eco_random_search, _build_rf_pipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data" / "Loan_default.csv"


def load_loan_default(path: Path):
    """
    Wczytuje Loan_default.csv, losuje stratified 20% wierszy
    i robi one-hot na cechach kategorycznych.
    """
    df_full = pd.read_csv(path)

    candidate_targets = [
        "Default", "default", "Loan_Default", "loan_default",
        "Loan Status", "loan_status"
    ]
    target_col = None
    for col in candidate_targets:
        if col in df_full.columns:
            target_col = col
            break
    if target_col is None:
        raise ValueError(
            f"Nie znalazłem kolumny z targetem. "
            f"Popraw candidate_targets w kodzie. Kolumny: {list(df_full.columns)}"
        )

    # losowe 20% z zachowaniem proporcji klas
    idx = np.arange(len(df_full))
    idx_sub, _ = train_test_split(
        idx,
        train_size=0.2,
        stratify=df_full[target_col],
        random_state=42,
    )
    df = df_full.iloc[idx_sub].reset_index(drop=True)

    y = df[target_col].values

    id_cols = [c for c in df.columns if "id" in c.lower()]
    X_df = df.drop(columns=[target_col] + id_cols)

    X_df = pd.get_dummies(X_df, drop_first=True)
    X = X_df.values

    logger.info(
        f"Wczytano dane (po subsamplingu 20%): X shape={X.shape}, y shape={y.shape}, "
        f"target={target_col}, usunięte ID kolumny={id_cols}"
    )
    return X, y, X_df.columns.tolist()


def compute_baseline_rf(
    X_train,
    y_train,
    X_test,
    y_test,
    fraction: float = 0.2,  # baseline: 20% danych
    random_state: int = 42,
):
    """
    Baseline: prosty RF na frakcji danych (domyślnie 20% trainu),
    bez optymalizacji hiperparametrów.
    Mierzymy:
      - AUC na teście
      - czas trenowania
      - emisje z CodeCarbon
    """

    frac = float(fraction)
    X_sub, _, y_sub, _ = train_test_split(
        X_train,
        y_train,
        train_size=frac,
        stratify=y_train,
        random_state=random_state,
    )

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    tracker = EmissionsTracker(
        project_name="baseline_rf",
        log_level="error",
        save_to_file=False,
    )

    tracker.start()
    t0 = time.time()
    model.fit(X_sub, y_sub)
    t1 = time.time()
    emissions_kg = tracker.stop()

    train_time_s = t1 - t0

    proba_test = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, proba_test)

    logger.info(
        f"[Baseline RF] frac={frac:.2f} AUC_test={auc_test:.3f} "
        f"emissions={emissions_kg:.2e} kg time={train_time_s:.2f}s"
    )

    return {
        "auc_test": auc_test,
        "emissions_kg": emissions_kg,
        "train_time_s": train_time_s,
    }


def classical_random_search_rf(
    X_train,
    y_train,
    X_test,
    y_test,
    n_iter: int = 30,
    random_state: int = 42,
):
    """
    Klasyczny RandomizedSearchCV na pełnych danych treningowych.
    Mierzymy emisje całego searcha i AUC na teście.
    """

    rf = RandomForestClassifier(random_state=random_state)

    param_distributions = {
        "clf__n_estimators": [100, 200, 300, 500],
        "clf__max_depth": [5, 10, 15, 20, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    }

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", rf),
    ])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )

    tracker = EmissionsTracker(
        project_name="classic_random_search_rf",
        log_level="error",
        save_to_file=False,
    )

    tracker.start()
    t0 = time.time()
    search.fit(X_train, y_train)
    t1 = time.time()
    emissions_kg = tracker.stop()

    search_time_s = t1 - t0

    best_model = search.best_estimator_
    proba_test = best_model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, proba_test)

    logger.info(f"[Classic RS] best AUC_test = {auc_test:.3f}")
    logger.info(f"[Classic RS] best params = {search.best_params_}")
    logger.info(
        f"[Classic RS] total emissions = {emissions_kg:.2e} kg CO2, "
        f"search_time={search_time_s:.2f}s"
    )

    return {
        "auc_test": auc_test,
        "emissions_kg": emissions_kg,      # emisje całego searcha
        "search_time_s": search_time_s,
        "best_params": search.best_params_,
    }


def train_final_rf_with_params(
    X_train,
    y_train,
    X_test,
    y_test,
    params: dict,
    project_name: str,
    random_state: int = 42,
):
    """
    Trening finalnego modelu RF na pełnym trainie z zadanymi parametrami.
    Mierzymy czas, emisje i AUC na teście.
    """

    model = _build_rf_pipeline(params, random_state=random_state)

    tracker = EmissionsTracker(
        project_name=project_name,
        log_level="error",
        save_to_file=False,
    )

    tracker.start()
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    emissions_kg = tracker.stop()

    train_time_s = t1 - t0

    proba_test = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, proba_test)

    logger.info(
        f"[Final {project_name}] AUC_test={auc_test:.3f}, "
        f"emissions={emissions_kg:.2e} kg, time={train_time_s:.2f}s"
    )

    return {
        "auc_test": auc_test,
        "emissions_kg": emissions_kg,
        "train_time_s": train_time_s,
    }


def run_experiment():
    # 1. dane + globalny train/test (ten sam dla baseline, eco i classic)
    X, y, feature_names = load_loan_default(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # 2. baseline RF na 20% trainu
    baseline = compute_baseline_rf(
        X_train,
        y_train,
        X_test,
        y_test,
        fraction=0.2,
        random_state=42,
    )
    baseline_auc = baseline["auc_test"]

    # 3. eco random search na jednej frakcji danych (np. 10% trainu)
    logger.info("=== Eco Random Search dla RF (EAUG + CodeCarbon) ===")
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    ECO_FRACTION = 0.10  # frakcja, na której działa eco search

    df_eco, best_eco = eco_random_search(
        X=X_train,
        y=y_train,
        param_distributions=param_distributions,
        data_fraction=ECO_FRACTION,
        baseline_auc=baseline_auc,
        n_iter=30,
        auc_min=baseline_auc - 0.02,
        random_state=123,
        project_name="eco_random_search_rf",
    )

    eco_search_total_emissions = df_eco["emissions_kg"].sum()
    eco_search_total_time = df_eco["train_time_s"].sum()

    logger.info(
        f"[Eco RS] best EAUG={best_eco['EAUG']:.2e}, "
        f"AUC_val={best_eco['roc_auc']:.3f}, "
        f"emissions(best)={best_eco['emissions_kg']:.2e} kg, "
        f"data_fraction={best_eco['data_fraction']:.2f}, "
        f"params={best_eco['params']}"
    )
    logger.info(
        f"[Eco RS] total search emissions={eco_search_total_emissions:.2e} kg, "
        f"total search time={eco_search_total_time:.2f}s"
    )

    # 4. finalny eco-model na pełnym trainie z best_eco params
    eco_final = train_final_rf_with_params(
        X_train,
        y_train,
        X_test,
        y_test,
        params=best_eco["params"],
        project_name="eco_final_rf",
        random_state=123,
    )

    # 5. klasyczny RandomizedSearchCV na pełnych danych
    logger.info("=== Klasyczny RandomizedSearch dla RF ===")
    classic = classical_random_search_rf(
        X_train,
        y_train,
        X_test,
        y_test,
        n_iter=30,
        random_state=42,
    )

    # zapis wyników eco-search do CSV (do wykresów punktowych)
    out_path = Path(__file__).parent / "data" / "eco_random_search_results.csv"
    df_eco.to_csv(out_path, index=False)
    logger.info(f"Zapisano wyniki eco_random_search do {out_path}")

    # 6. EAUG dla trzech wariantów (liczone na globalnym teście)

    def calc_eaug(auc_test, baseline_auc_, emissions_kg_):
        delta = max(0.0, auc_test - baseline_auc_)
        return delta / emissions_kg_ if emissions_kg_ > 0 else 0.0

    # baseline – brak zysku ponad siebie samego
    eaug_baseline = 0.0

    # EAUG tylko z emisji finalnego treningu eco-modelu
    eaug_eco_final_only = calc_eaug(
        eco_final["auc_test"],
        baseline_auc,
        eco_final["emissions_kg"],
    )

    # EAUG z emisji całego procesu: search + finalny trening
    eaug_eco_full = calc_eaug(
        eco_final["auc_test"],
        baseline_auc,
        eco_search_total_emissions + eco_final["emissions_kg"],
    )

    # dla klasycznego RS mamy emisje całego searcha (w tym finalnego modelu)
    eaug_classic = calc_eaug(
        classic["auc_test"],
        baseline_auc,
        classic["emissions_kg"],
    )

    # 7. PODSUMOWANIE + logowa „tabelka”
    logger.info("=== PODSUMOWANIE ===")
    logger.info(
        f"Baseline RF (20% train) -> "
        f"AUC_test={baseline_auc:.3f}, "
        f"emissions={baseline['emissions_kg']:.2e} kg, "
        f"time={baseline['train_time_s']:.2f}s, "
        f"EAUG={eaug_baseline:.2e}"
    )
    logger.info(
        f"Eco RS final RF (fraction={ECO_FRACTION:.2f}) -> "
        f"AUC_test={eco_final['auc_test']:.3f}, "
        f"search_emissions={eco_search_total_emissions:.2e} kg, "
        f"search_time={eco_search_total_time:.2f}s, "
        f"final_emissions={eco_final['emissions_kg']:.2e} kg, "
        f"final_time={eco_final['train_time_s']:.2f}s, "
        f"EAUG_final_only={eaug_eco_final_only:.2e}, "
        f"EAUG_full={eaug_eco_full:.2e}"
    )
    logger.info(
        f"Classic RS final RF -> "
        f"AUC_test={classic['auc_test']:.3f}, "
        f"search_emissions={classic['emissions_kg']:.2e} kg, "
        f"search_time={classic['search_time_s']:.2f}s, "
        f"EAUG={eaug_classic:.2e}"
    )

    # 8. TABELKA PORÓWNAWCZA -> CSV
    summary_rows = [
        {
            "method": "baseline_rf_20pct",
            "auc_test": baseline_auc,
            "train_time_s": baseline["train_time_s"],
            "emissions_train_kg": baseline["emissions_kg"],
            "search_time_s": 0.0,
            "emissions_search_kg": 0.0,
            "EAUG_final_only": eaug_baseline,
            "EAUG_full": eaug_baseline,
        },
        {
            "method": f"eco_rs_fraction_{ECO_FRACTION:.2f}",
            "auc_test": eco_final["auc_test"],
            "train_time_s": eco_final["train_time_s"],
            "emissions_train_kg": eco_final["emissions_kg"],
            "search_time_s": eco_search_total_time,
            "emissions_search_kg": eco_search_total_emissions,
            "EAUG_final_only": eaug_eco_final_only,
            "EAUG_full": eaug_eco_full,
        },
        {
            "method": "classic_random_search",
            "auc_test": classic["auc_test"],
            "train_time_s": np.nan,  # wliczone w search_time
            "emissions_train_kg": np.nan,  # wliczone w emissions_search_kg
            "search_time_s": classic["search_time_s"],
            "emissions_search_kg": classic["emissions_kg"],
            "EAUG_final_only": eaug_classic,
            "EAUG_full": eaug_classic,
        },
    ]

    df_summary = pd.DataFrame(summary_rows)
    summary_path = Path(__file__).parent / "data" / "experiment_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"Zapisano podsumowanie eksperymentu do {summary_path}")


if __name__ == "__main__":
    run_experiment()
