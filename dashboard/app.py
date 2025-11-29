# dashboard/app.py

import streamlit as st
import pandas as pd
import altair as alt
import pickle
from pathlib import Path
import numpy as np

# ==========================
# KONFIGURACJA STRONY
# ==========================
st.set_page_config(
    page_title="EcoSearch – ekologiczne strojenie modeli",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 EcoSearch – ekologiczne strojenie modeli")
st.write(
    """
    Ten panel pomaga dobrać **model ML**, który spełnia zadane oczekiwania jakości (AUC),
    jednocześnie **minimalizując ślad węglowy** i czas trenowania.
    """
)

# ==========================
# IMPORT WASZYCH FUNKCJI
# ==========================
# TODO: DOSTOSUJ TE IMPORTY DO SWOJEGO KODU
# Przykład:
# from eco_search import eco_random_search as real_eco_random_search
# from utils import load_dataset as real_load_dataset


def load_dataset(name: str):
    """
    Tymczasowy loader – ZASTĄP własną funkcją z utils.load_datasets().
    Powinien zwracać (X, y).
    """
    # Przykład:
    # from utils import load_datasets
    # return load_datasets(name)

    # Placeholder: pusty dataframe – po podmianie nie będzie potrzebny.
    X = pd.DataFrame()
    y = pd.Series(dtype=float)
    return X, y


def load_meta_model(path: str = "models/meta_emissions_model.pkl"):
    """
    Ładuje meta-model z pliku .pkl (np. do predykcji emisji / czasu).
    Jeśli eco_random_search sam go wczytuje, możesz tego nie używać.
    """
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        model = pickle.load(f)
    return model


# TODO: Podmień na prawdziwy import
def eco_random_search(
    X,
    y,
    model_family: str,
    max_fits: int,
    min_auc: float,
    max_data_fraction: float,
    fractions_list,
    meta_model=None,
):
    """
    Placeholder – ZASTĄP własną implementacją eco_random_search(...).

    Tu robimy sensowniejszą symulację:
    - losujemy konfiguracje z różnych frakcji danych,
    - AUC i emisje zależą od frakcji,
    - wyniki ZMIENIAJĄ się przy zmianie fractions_list.

    Zwracamy DataFrame z kolumnami:
    - 'auc'
    - 'emissions'
    - 'eaug'
    - 'config'
    """

    if not fractions_list:
        fractions_list = [max_data_fraction]

    # Seed zależny od parametrów → stabilne dla danych ustawień,
    # ale inne dla różnych zestawów frakcji / modeli.
    seed = abs(
        hash((model_family, max_fits, min_auc, max_data_fraction, tuple(sorted(fractions_list))))
    ) % (2**32)
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(max_fits):
        frac = float(rng.choice(fractions_list))

        # "intuicja": najlepsze AUC zwykle przy większych frakcjach,
        # ale z pewnym szumem.
        base_auc = min_auc + 0.15 * frac  # rośnie z frakcją
        auc = base_auc + 0.03 * rng.normal()  # trochę szumu
        auc = float(np.clip(auc, min_auc, 0.99))

        # Emisje ~ rosną z frakcją danych i trochę złożonością modelu
        model_complexity = {
            "log_reg": 1.0,
            "random_forest": 1.8,
            "mlp": 2.2,
        }.get(model_family, 1.5)

        emissions = (5 + 30 * frac) * model_complexity * (0.7 + 0.6 * rng.random())
        emissions = float(max(emissions, 1e-3))

        eaug = auc / emissions

        config = {
            "data_fraction": frac,
            "random_seed": int(rng.integers(0, 10_000)),
            "model_complexity": model_complexity,
        }

        rows.append(
            {
                "auc": auc,
                "emissions": emissions,
                "eaug": eaug,
                "config": config,
            }
        )

    df = pd.DataFrame(rows)
    df = df[df["auc"] >= min_auc].reset_index(drop=True)
    return df


# ==========================
# CACHE – ładowanie danych i meta-modelu
# ==========================

@st.cache_data
def cached_load_dataset(name: str):
    return load_dataset(name)


@st.cache_resource
def cached_meta_model(path: str):
    return load_meta_model(path)


meta_model = cached_meta_model("models/meta_emissions_model.pkl")

# ==========================
# SIDEBAR – USTAWIENIA UŻYTKOWNIKA
# ==========================

with st.sidebar:
    st.header("⚙️ Ustawienia EcoSearch")

    dataset = st.selectbox(
        "Zbiór danych",
        options=["titanic", "penguins"],
        index=0,
    )

    model_family = st.selectbox(
        "Rodzina modelu",
        options=[
            "log_reg",
            "random_forest",
            "mlp",
        ],
        format_func=lambda x: {
            "log_reg": "Logistic Regression",
            "random_forest": "Random Forest",
            "mlp": "MLP (sieć neuronowa)",
        }.get(x, x),
    )

    max_fits = st.slider(
        "Maksymalna liczba trenowań modeli",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
    )

    min_auc = st.slider(
        "Minimalnie akceptowalne AUC",
        min_value=0.60,
        max_value=0.95,
        value=0.75,
        step=0.01,
    )

    st.markdown("**Frakcje danych** wykorzystywane przez EcoSearch:")

    fractions_preset = [0.1, 0.25, 0.5, 0.75, 1.0]
    selected_fractions = st.multiselect(
        "Wybierz frakcje danych",
        options=fractions_preset,
        default=[0.25, 0.5, 0.75],
    )

    if not selected_fractions:
        st.warning("Wybierz chociaż jedną frakcję danych – ustawiam domyślnie 0.5.")
        selected_fractions = [0.5]

    max_data_fraction = max(selected_fractions)

    st.write(f"**Maks. frakcja danych:** {max_data_fraction:.2f}")

    run_button = st.button("🚀 Run eco search", type="primary")

# ==========================
# GŁÓWNA LOGIKA – URUCHOMIENIE SEARCHA
# ==========================

if run_button:
    st.subheader("🔄 Uruchamianie EcoSearch…")

    X, y = cached_load_dataset(dataset)

    with st.spinner("Optymalizuję konfiguracje pod kątem AUC i emisji CO₂..."):
        results_df = eco_random_search(
            X=X,
            y=y,
            model_family=model_family,
            max_fits=max_fits,
            min_auc=min_auc,
            max_data_fraction=max_data_fraction,
            fractions_list=selected_fractions,
            meta_model=meta_model,
        )

    if results_df is None or len(results_df) == 0:
        st.error("EcoSearch nie zwrócił żadnych konfiguracji. Sprawdź parametry lub implementację.")
        if "results" in st.session_state:
            del st.session_state["results"]
    else:
        # Zapisujemy wyniki w session_state, żeby były dostępne
        st.session_state["results"] = {
            "results_df": results_df,
            "dataset": dataset,
            "model_family": model_family,
            "max_fits": max_fits,
            "min_auc": min_auc,
            "selected_fractions": selected_fractions,
        }

# ==========================
# PREZENTACJA WYNIKÓW (także po kliknięciu przycisku raportu)
# ==========================

if "results" in st.session_state:
    data_state = st.session_state["results"]
    results_df = data_state["results_df"]
    dataset = data_state["dataset"]
    model_family = data_state["model_family"]
    max_fits = data_state["max_fits"]
    min_auc = data_state["min_auc"]
    selected_fractions = data_state["selected_fractions"]

    auc_col = "auc"
    emissions_col = "emissions"
    eaug_col = "eaug"

    # 1. Najlepsza konfiguracja
    st.markdown("---")
    st.subheader("🥇 Najlepsza znaleziona konfiguracja")

    best_idx = results_df[eaug_col].idxmax()
    best_row = results_df.loc[best_idx]
    config = best_row.get("config", {})

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        st.metric("AUC", f"{best_row[auc_col]:.3f}")

    with col2:
        st.metric("Szacowane emisje CO₂ [jedn.]", f"{best_row[emissions_col]:.2f}")

    with col3:
        st.metric("EAUG (jakość / emisje)", f"{best_row[eaug_col]:.4f}")

    with col4:
        st.markdown(
            """
            **🌿 Zielona ocena:**  
            Im wyższy EAUG, tym **lepsza jakość przy niższych emisjach**.
            """
        )

    st.markdown("**Najważniejsze hiperparametry:**")
    if isinstance(config, dict):
        config_df = pd.DataFrame(
            {"Hiperparametr": list(config.keys()), "Wartość": list(config.values())}
        )
        st.table(config_df)
    else:
        st.write(config)

    # 2. TOP N konfiguracji
    st.markdown("---")
    st.subheader("📊 TOP konfiguracje wg EAUG")

    top_n = min(10, len(results_df))
    top_df = results_df.sort_values(by=eaug_col, ascending=False).head(top_n).copy()

    for c in [auc_col, emissions_col, eaug_col]:
        if c in top_df.columns:
            top_df[c] = top_df[c].astype(float).round(4)

    st.dataframe(top_df, use_container_width=True)

    # 3. Wykres: AUC vs emisje
    st.markdown("---")
    st.subheader("🌍 Wykres AUC vs emisje CO₂")

    plot_df = results_df.copy()
    plot_df["config_str"] = plot_df["config"].astype(str)

    chart = (
        alt.Chart(plot_df)
        .mark_circle(size=100, opacity=0.8)
        .encode(
            x=alt.X(auc_col, title="AUC"),
            y=alt.Y(emissions_col, title="Szacowane emisje CO₂"),
            color=alt.Color(eaug_col, title="EAUG"),
            tooltip=[
                auc_col,
                emissions_col,
                eaug_col,
                "config_str",
            ],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    # 4. RAPORT TEKSTOWY
    st.markdown("---")
    st.subheader("📄 Raport dla klienta (wersja tekstowa)")

    def build_text_report(
        dataset,
        model_family,
        max_fits,
        min_auc,
        selected_fractions,
        best_row,
        auc_col,
        emissions_col,
        eaug_col,
        config,
    ):
        lines = []
        lines.append("EcoSearch – raport z ekologicznego strojenia modelu\n")
        lines.append(f"Zbiór danych: {dataset}")
        lines.append(f"Rodzina modelu: {model_family}")
        lines.append(f"Maksymalna liczba trenowań: {max_fits}")
        lines.append(f"Minimalne akceptowane AUC: {min_auc:.3f}")
        lines.append(f"Frakcje danych: {', '.join(str(f) for f in selected_fractions)}\n")

        lines.append("== Najlepsza konfiguracja ==\n")
        lines.append(f"AUC: {best_row[auc_col]:.4f}")
        lines.append(f"Szacowane emisje CO₂: {best_row[emissions_col]:.4f}")
        lines.append(f"EAUG (jakość / emisje): {best_row[eaug_col]:.6f}\n")

        lines.append("Hiperparametry:")
        if isinstance(config, dict):
            for k, v in config.items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append(str(config))
        lines.append("\n")

        lines.append("== Podsumowanie ==")
        lines.append(
            "EcoSearch odnalazł konfigurację, która spełnia wymagania jakościowe "
            "przy możliwie niskich emisjach CO₂ (wysoki EAUG)."
        )

        return "\n".join(lines)

    generate_report = st.button("🧾 Generate PDF / text report")

    if generate_report:
        report_text = build_text_report(
            dataset=dataset,
            model_family=model_family,
            max_fits=max_fits,
            min_auc=min_auc,
            selected_fractions=selected_fractions,
            best_row=best_row,
            auc_col=auc_col,
            emissions_col=emissions_col,
            eaug_col=eaug_col,
            config=config,
        )

        st.text_area("Podgląd raportu tekstowego", value=report_text, height=300)

        st.download_button(
            label="⬇️ Pobierz raport jako .txt",
            data=report_text,
            file_name="ecosearch_report.txt",
            mime="text/plain",
        )

else:
    st.info("Skonfiguruj parametry w panelu po lewej i kliknij **🚀 Run eco search**.")
