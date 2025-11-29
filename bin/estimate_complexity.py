import ast
import logging

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("estimate_complexity.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


INPUT_PATH = "real_ml_energy_dataset_random_balanced.csv"
OUTPUT_PATH = "meta_ml_energy_dataset_random_balanced.csv"


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def parse_params(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Parsing params column into Python dict")
    df["params_dict"] = df["params"].apply(ast.literal_eval)
    return df


def extract_param_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Extracting parameter features")

    def _extract(row):
        fam = row["model_family"]
        p = row["params_dict"]

        out = {
            "lr_C": np.nan,
            "lr_max_iter": np.nan,
            "rf_n_estimators": np.nan,
            "rf_max_depth": np.nan,
            "mlp_hidden_total": np.nan,
            "mlp_hidden_layers": np.nan,
            "mlp_max_iter": np.nan,
        }

        if fam == "log_reg":
            out["lr_C"] = p.get("C", np.nan)
            out["lr_max_iter"] = p.get("max_iter", np.nan)

        elif fam == "random_forest":
            out["rf_n_estimators"] = p.get("n_estimators", np.nan)
            out["rf_max_depth"] = p.get("max_depth", np.nan)

        elif fam == "mlp":
            h = p.get("hidden_layer_sizes", ())
            if isinstance(h, (int, float)):
                h = (int(h),)
            if not isinstance(h, (tuple, list)):
                h = tuple()
            out["mlp_hidden_total"] = float(sum(h)) if len(h) > 0 else np.nan
            out["mlp_hidden_layers"] = len(h) if len(h) > 0 else np.nan
            out["mlp_max_iter"] = p.get("max_iter", np.nan)

        return pd.Series(out)

    feat = df.apply(_extract, axis=1)
    df = pd.concat([df, feat], axis=1)
    return df


def estimate_complexity(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing complexity_score")

    def _complexity(row):
        fam = row["model_family"]
        n_features = row.get("n_features", np.nan)
        p = row["params_dict"]

        if fam == "log_reg":
            return float(n_features)

        if fam == "random_forest":
            n_estimators = p.get("n_estimators", 100)
            max_depth = p.get("max_depth", 10)
            if max_depth is None:
                max_depth = 10
            return float(n_estimators) * float(max_depth)

        if fam == "mlp":
            h = p.get("hidden_layer_sizes", ())
            if isinstance(h, (int, float)):
                h = (int(h),)
            if not isinstance(h, (tuple, list)):
                h = tuple()
            if len(h) == 0:
                return np.nan
            total_params = n_features * h[0]
            for i in range(len(h) - 1):
                total_params += h[i] * h[i + 1]
            return float(total_params)

        return np.nan

    df["complexity_score"] = df.apply(_complexity, axis=1)
    return df


def compute_emission_aware_auc_gain(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing Emission-aware AUC gain (EAUG)")

    # baseline per dataset: średni AUC dla danego zbioru
    df["baseline_auc"] = df.groupby("dataset_name")["roc_auc"].transform("mean")

    # ΔAUC (zysk ponad baseline, ale tylko dodatni)
    df["delta_auc"] = (df["roc_auc"] - df["baseline_auc"]).clip(lower=0)

    # EAUG = delta_auc / emissions
    # zabezpieczamy się przed dzieleniem przez 0
    df["EAUG"] = df.apply(
        lambda row: row["delta_auc"] / row["emissions_kg"] if row["emissions_kg"] > 0 else np.nan,
        axis=1,
    )

    return df


def main():
    df = load_data(INPUT_PATH)
    df = parse_params(df)
    df = extract_param_features(df)
    df = estimate_complexity(df)
    df = compute_emission_aware_auc_gain(df)

    logger.info(f"Saving processed data to {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Done")


if __name__ == "__main__":
    main()
