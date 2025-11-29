import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

GREEN = "#23CF71"

def add_value_labels(ax, fmt="%.3f"):
    for p in ax.patches:
        value = p.get_width()
        ax.annotate(
            fmt % value,
            (p.get_width(), p.get_y() + p.get_height() / 2),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=11,
        )

def main():
    base = Path(__file__).parent
    data_path = base / "data" / "experiment_summary.csv"
    out_dir = base / "plots"
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)

    df["total_time_s"] = df[["train_time_s", "search_time_s"]].sum(axis=1, skipna=True)
    df["total_emissions_kg"] = df[["emissions_train_kg", "emissions_search_kg"]].sum(axis=1, skipna=True)

    def prettify(method):
        if method.startswith("eco_rs_fraction_"):
            frac = method.replace("eco_rs_fraction_", "")
            return f"Eco RS (frac={frac})"
        if method == "baseline_rf_20pct":
            return "Baseline RF (20%)"
        if method == "classic_random_search":
            return "Classic RS"
        return method

    df["method_label"] = df["method"].map(prettify)
    baseline_auc = df.loc[df["method_label"] == "Baseline RF (20%)", "auc_test"].iloc[0]

    eco_label = df.loc[df["method_label"].str.startswith("Eco RS"), "method_label"].iloc[0]
    classic_label = "Classic RS"
    order_no_base = [eco_label, classic_label]

    sns.set_theme(style="whitegrid")

    df_auc = df.copy().sort_values("auc_test", ascending=True)
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(
        data=df_auc,
        y="method_label",
        x="auc_test",
        orient="h",
        color=GREEN,
        edgecolor="black"
    )
    plt.xlabel("AUC (test)")
    plt.ylabel("")
    plt.title("Porównanie jakości modeli (AUC test)")
    plt.xlim(0.5, 1.0)
    plt.axvline(baseline_auc, color="red", linestyle="--", label=f"Baseline = {baseline_auc:.3f}")
    plt.legend()
    add_value_labels(ax, fmt="%.3f")
    plt.tight_layout()
    plt.savefig(out_dir / "auc_test.png", dpi=200, transparent=True)

    df_no_base = df[df["method_label"] != "Baseline RF (20%)"]

    df_eaug = df_no_base.sort_values("EAUG_full", ascending=True)
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(
        data=df_eaug,
        y="method_label",
        x="EAUG_full",
        orient="h",
        color=GREEN,
        order=order_no_base,
        edgecolor="black"
    )
    plt.xlabel("EAUG_full (ΔAUC / CO2)")
    plt.ylabel("")
    plt.title("Efektywność emisyjna (EAUG_full)")
    add_value_labels(ax, fmt="%.2f")
    plt.tight_layout()
    plt.savefig(out_dir / "eaug_full.png", dpi=200, transparent=True)

    df_time = df_no_base.sort_values("total_time_s", ascending=True)
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(
        data=df_time,
        y="method_label",
        x="total_time_s",
        orient="h",
        color=GREEN,
        order=order_no_base,
        edgecolor="black"
    )
    plt.xlabel("Czas całkowity [s]")
    plt.ylabel("")
    plt.title("Czas obliczeń (search + trening)")
    add_value_labels(ax, fmt="%.1f")
    plt.tight_layout()
    plt.savefig(out_dir / "total_time.png", dpi=200, transparent=True)

    df_em = df_no_base.sort_values("total_emissions_kg", ascending=True)
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(
        data=df_em,
        y="method_label",
        x="total_emissions_kg",
        orient="h",
        color=GREEN,
        order=order_no_base,
        edgecolor="black"
    )
    plt.xlabel("Emisje CO2 [kg]")
    plt.ylabel("")
    plt.title("Emisje CO2 (search + trening)")
    add_value_labels(ax, fmt="%.5f")
    plt.tight_layout()
    plt.savefig(out_dir / "total_emissions.png", dpi=200, transparent=True)

    print("\nZapisano wykresy w folderze 'plots/':")
    print(" plots/auc_test.png")
    print(" plots/eaug_full.png")
    print(" plots/total_time.png")
    print(" plots/total_emissions.png")


if __name__ == "__main__":
    main()
