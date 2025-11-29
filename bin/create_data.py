import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_datasets, sample_random_config, run_experiment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("real_training.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    out_dir = Path("codecarbon_logs")
    out_dir.mkdir(exist_ok=True)

    datasets = load_datasets()
    rng = np.random.default_rng(42)

    families = ["log_reg", "random_forest", "mlp"]
    n_runs_per_family = 40  # na każdy dataset i każdy typ modelu

    results = []
    exp_id = 0

    for name, (X, y) in datasets.items():
        logger.info(f"Dataset {name}: {n_runs_per_family} runs per family")
        for family in families:
            for _ in range(n_runs_per_family):
                exp_id += 1
                params = sample_random_config(family, rng)
                seed = int(rng.integers(0, 1_000_000))
                data_fraction = rng.uniform(0.2, 1.0)

                try:
                    res = run_experiment(
                        dataset_name=name,
                        X=X,
                        y=y,
                        family=family,
                        params=params,
                        exp_id=exp_id,
                        seed=seed,
                        data_fraction=data_fraction,
                        output_dir=out_dir,
                        logger=logger,
                    )
                    results.append(res)
                except Exception as e:
                    logger.error(f"[{exp_id}] {name} {family} seed={seed} ERROR: {e}")

    df = pd.DataFrame(results)
    df.to_csv("real_ml_energy_dataset_random_balanced.csv", index=False)
    logger.info(f"Saved {len(df)} rows to real_ml_energy_dataset_random_balanced.csv")


if __name__ == "__main__":
    logger.info("START")
    main()
    logger.info("END")
