import time
import logging
from itertools import product
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from codecarbon import EmissionsTracker


logger = logging.getLogger(__name__)


def _build_rf_pipeline(params: Dict[str, Any], random_state: int) -> Pipeline:
    clf = RandomForestClassifier(random_state=random_state, **params)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # bezpieczne dla one-hot
        ("clf", clf),
    ])
    return pipe


def _evaluate_config(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    data_fraction: float,
    baseline_auc: float,
    random_state: int = 42,
    project_name: str = "eco_random_search",
) -> Dict[str, Any]:
    """
    Trenuje RF na zadanej frakcji danych (data_fraction) i liczy:
    - AUC na walidacji
    - czas trenowania
    - emisje CO2 z CodeCarbon (kg)
    - EAUG = max(0, AUC - baseline_auc) / emissions_kg
    """

    frac = float(data_fraction)
    assert 0 < frac <= 1.0

    # podzbiór danych
    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=frac,
        stratify=y,
        random_state=random_state,
    )

    # train / val w obrębie frakcji
    X_train, X_val, y_train, y_val = train_test_split(
        X_sub,
        y_sub,
        test_size=0.2,
        stratify=y_sub,
        random_state=random_state,
    )

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

    proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)

    delta_auc = max(0.0, auc - baseline_auc)
    eaug = delta_auc / emissions_kg if emissions_kg > 0 and delta_auc > 0 else 0.0

    result = {
        "params": params,
        "data_fraction": frac,
        "n_samples_sub": X_sub.shape[0],
        "train_time_s": train_time_s,
        "emissions_kg": emissions_kg,
        "roc_auc": auc,
        "baseline_auc": baseline_auc,
        "delta_auc": delta_auc,
        "EAUG": eaug,
    }
    return result


def _sample_from_distributions(
    param_distributions: Dict[str, List[Any]],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    sampled = {}
    for name, values in param_distributions.items():
        sampled[name] = rng.choice(values)
    return sampled


def eco_random_search(
    X: np.ndarray,
    y: np.ndarray,
    param_distributions: Dict[str, List[Any]],
    data_fraction: float,
    baseline_auc: float,
    n_iter: int = 30,
    auc_min: Optional[float] = None,
    random_state: int = 42,
    project_name: str = "eco_random_search",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ekologiczny random search dla RF:
    - losuje tylko hiperparametry (n_iter razy)
    - ZAWSZE używa tej samej frakcji danych (data_fraction)
    - mierzy emisje CodeCarbon przy każdym fitcie
    - wybiera konfigurację maksymalizującą EAUG
    """

    rng = np.random.default_rng(random_state)
    results = []

    for i in range(n_iter):
        params = _sample_from_distributions(param_distributions, rng)
        seed = int(rng.integers(0, 1_000_000))

        res = _evaluate_config(
            X=X,
            y=y,
            params=params,
            data_fraction=data_fraction,
            baseline_auc=baseline_auc,
            random_state=seed,
            project_name=project_name,
        )
        res["iter"] = i
        results.append(res)
        logger.info(
            f"[eco_random_search] iter={i} frac={data_fraction:.2f} "
            f"AUC={res['roc_auc']:.3f} EAUG={res['EAUG']:.2e} "
            f"emissions={res['emissions_kg']:.2e} kg "
            f"time={res['train_time_s']:.2f}s"
        )

    df = pd.DataFrame(results)

    if auc_min is not None:
        mask = df["roc_auc"] >= auc_min
        df_candidate = df[mask] if mask.any() else df
    else:
        df_candidate = df

    best_idx = df_candidate["EAUG"].idxmax()
    best_config = df_candidate.loc[best_idx].to_dict()

    return df, best_config


def eco_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List[Any]],
    data_fraction: float,
    baseline_auc: float,
    max_evals: Optional[int] = None,
    auc_min: Optional[float] = None,
    random_state: int = 42,
    project_name: str = "eco_grid_search",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ekologiczny grid search:
    - iteruje po siatce hiperparametrów
    - ZAWSZE używa tej samej frakcji danych (data_fraction)
    - mierzy emisje CodeCarbon
    - wybiera konfigurację z max EAUG
    """

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = list(product(*values))
    rng = np.random.default_rng(random_state)

    results = []
    eval_count = 0

    for combo in combos:
        params = dict(zip(keys, combo))
        if max_evals is not None and eval_count >= max_evals:
            break

        seed = int(rng.integers(0, 1_000_000))
        res = _evaluate_config(
            X=X,
            y=y,
            params=params,
            data_fraction=data_fraction,
            baseline_auc=baseline_auc,
            random_state=seed,
            project_name=project_name,
        )
        res["eval_id"] = eval_count
        results.append(res)
        eval_count += 1

        logger.info(
            f"[eco_grid_search] eval={eval_count} frac={data_fraction:.2f} "
            f"AUC={res['roc_auc']:.3f} EAUG={res['EAUG']:.2e} "
            f"emissions={res['emissions_kg']:.2e} kg "
            f"time={res['train_time_s']:.2f}s"
        )

    df = pd.DataFrame(results)

    if auc_min is not None:
        mask = df["roc_auc"] >= auc_min
        df_candidate = df[mask] if mask.any() else df
    else:
        df_candidate = df

    best_idx = df_candidate["EAUG"].idxmax()
    best_config = df_candidate.loc[best_idx].to_dict()

    return df, best_config
