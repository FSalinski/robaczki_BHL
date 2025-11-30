import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import numpy as np
import sys
from sklearn.datasets import make_classification

# =========================================
# ŚCIEŻKI I IMPORT ECO_SEARCH Z robaczki_BHL/src
# =========================================
ROOT = Path(__file__).resolve().parent.parent  # robaczki_BHL
SRC_PATH = ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from eco_search import eco_random_search  # z src/eco_search.py

# =========================================
# KONFIGURACJA STRONY I MOTYW
# =========================================
st.set_page_config(
    page_title="EcoSearch – ekologiczne strojenie modeli",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #062820;
        color: #f5f5f5;
    }
    [data-testid="stSidebar"] {
        background-color: #07352a;
    }
    section.main > div {
        background-color: #0b3d33;
        padding: 1rem 1.5rem 2rem 1.5rem;
        border-radius: 0.75rem;
    }
    div.stButton > button {
        background-color: #146c43;
        color: white;
        border-radius: 999px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #0f5132;
        color: white;
    }
    .stMetric {
        background-color: #0b3d33;
        padding: 0.5rem;
        border-radius: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌿 EcoSearch – ekologiczne strojenie modeli")
st.write(
    """
    Ten panel pomaga dobrać **model ML**, który spełnia zadane oczekiwania jakości (AUC),
    jednocześnie **minimalizując ślad węglowy** i czas trenowania.
    """
)

# =========================================
# ŁADOWANIE DANYCH (syntetyczne dla demo)
# =========================================
def load_dataset(name: str):
    seed = abs(hash(name)) % (2**32)
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=seed,
    )
    cols = [f"x{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=cols)
    y_ser = pd.Series(y, name="target")
    return X_df, y_ser


@st.cache_data
def cached_load_dataset(name: str):
    return load_dataset(name)


# =========================================
# PARAM_DISTRIBUTIONS DLA RF
# =========================================
def get_param_distributions(model_family: str):
    return {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }


# =========================================
# SIDEBAR – USTAWIENIA
# =========================================
with st.sidebar:
    st.header("⚙️ Ustawienia EcoSearch")

    data_source = st.radio(
        "Źródło danych",
        options=["Wbudowane zbiory", "Własny plik (CSV)"],
        index=0,
    )

    if data_source == "Wbudowane zbiory":
        dataset = st.selectbox(
            "Zbiór danych",
            options=["titanic", "penguins"],
            index=0,
        )
    else:
        dataset = None

    model_family = st.selectbox(
        "Rodzina modelu",
        options=["log_reg", "random_forest", "mlp"],
        format_func=lambda x: {
            "log_reg": "Logistic Regression",
            "random_forest": "Random Forest",
            "mlp": "MLP (sieć neuronowa)",
        }.get(x, x),
    )

    max_fits = st.slider(
        "Maksymalna liczba trenowań modeli (łącznie)",
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


# =========================================
# GŁÓWNA CZĘŚĆ – WŁASNY PLIK CSV
# =========================================
custom_target_col = None

if data_source == "Własny plik (CSV)":
    st.markdown("---")
    st.subheader("📁 Własny zbiór danych (CSV)")

    uploaded_file = st.file_uploader(
        "Przeciągnij tutaj plik CSV lub kliknij, aby wybrać",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Nie udało się wczytać pliku: {e}")
            df = None

        if df is not None:
            st.write("Podgląd danych (pierwsze 5 wierszy):")
            st.dataframe(df.head(), use_container_width=True)

            all_cols = list(df.columns)
            if all_cols:
                custom_target_col = st.selectbox(
                    "Wybierz kolumnę celu (y)",
                    options=all_cols,
                )
                if custom_target_col:
                    st.session_state["custom_data"] = {
                        "df": df,
                        "target_col": custom_target_col,
                        "file_name": uploaded_file.name,
                    }
            else:
                st.error("Plik nie zawiera żadnych kolumn.")
    else:
        st.info("Przeciągnij plik CSV w pole powyżej, aby go użyć jako zbiór danych.")


# =========================================
# URUCHOMIENIE ECO RANDOM SEARCH
# =========================================
if run_button:
    st.subheader("🔄 Uruchamianie EcoSearch…")

    if data_source == "Wbudowane zbiory":
        X_df, y_ser = cached_load_dataset(dataset)
        used_dataset_name = dataset
    else:
        if "custom_data" not in st.session_state:
            st.error("Najpierw wgraj plik CSV i wybierz kolumnę celu.")
            if "results" in st.session_state:
                del st.session_state["results"]
            X_df = y_ser = None
        else:
            custom = st.session_state["custom_data"]
            df = custom["df"]
            target_col = custom["target_col"]

            if target_col not in df.columns:
                st.error("Wybrana kolumna celu nie istnieje w danych. Wgraj plik ponownie.")
                if "results" in st.session_state:
                    del st.session_state["results"]
                X_df = y_ser = None
            else:
                y_ser = df[target_col]
                X_df = df.drop(columns=[target_col])

                numeric_cols = X_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    st.error(
                        "W wgranym pliku nie ma żadnych kolumn numerycznych po odjęciu kolumny celu. "
                        "Dodaj numeryczne cechy albo przygotuj dane wcześniej."
                    )
                    if "results" in st.session_state:
                        del st.session_state["results"]
                    X_df = y_ser = None
                else:
                    if len(numeric_cols) < X_df.shape[1]:
                        dropped = set(X_df.columns) - set(numeric_cols)
                        st.warning(
                            "Z pliku usunięto nienumeryczne kolumny cech, "
                            f"np.: {', '.join(list(dropped)[:5])}"
                        )
                    X_df = X_df[numeric_cols]
                    used_dataset_name = f"Własny plik: {custom.get('file_name', 'dataset.csv')}"

    if X_df is not None and y_ser is not None:
        if len(X_df) == 0 or len(y_ser) == 0:
            st.error(
                "Zbiór danych ma 0 obserwacji – nie można uruchomić EcoSearch. "
                "Sprawdź loader danych lub plik CSV."
            )
            if "results" in st.session_state:
                del st.session_state["results"]
        else:
            X = X_df.values
            y = y_ser.values

            param_distributions = get_param_distributions(model_family)
            baseline_auc = float(min_auc)

            results_list = []
            random_state_base = 42
            n_fractions = len(selected_fractions)
            n_iter_per_frac = max(1, max_fits // n_fractions)

            with st.spinner("Optymalizuję konfiguracje pod kątem AUC i emisji CO₂..."):
                for i, frac in enumerate(selected_fractions):
                    rs = random_state_base + i * 1000
                    df_frac, _best_cfg = eco_random_search(
                        X=X,
                        y=y,
                        param_distributions=param_distributions,
                        data_fraction=float(frac),
                        baseline_auc=baseline_auc,
                        n_iter=int(n_iter_per_frac),
                        auc_min=min_auc,
                        random_state=rs,
                        project_name=f"eco_random_search_frac_{frac}",
                    )
                    df_frac["data_fraction"] = float(frac)
                    results_list.append(df_frac)

            if not results_list:
                st.error("EcoSearch nie zwrócił żadnych konfiguracji.")
                if "results" in st.session_state:
                    del st.session_state["results"]
            else:
                results_df = pd.concat(results_list, ignore_index=True)
                st.session_state["results"] = {
                    "results_df": results_df,
                    "dataset": used_dataset_name,
                    "model_family": model_family,
                    "max_fits": max_fits,
                    "min_auc": min_auc,
                    "selected_fractions": selected_fractions,
                }

# =========================================
# PREZENTACJA WYNIKÓW I RAPORT
# =========================================
if "results" in st.session_state:
    data_state = st.session_state["results"]
    results_df = data_state["results_df"]
    dataset_name = data_state["dataset"]
    model_family = data_state["model_family"]
    max_fits = data_state["max_fits"]
    min_auc = data_state["min_auc"]
    selected_fractions = data_state["selected_fractions"]

    auc_col = "roc_auc"
    emissions_col = "emissions_kg"
    eaug_col = "EAUG"

    st.markdown("---")
    st.subheader("🥇 Najlepsza znaleziona konfiguracja")

    best_idx = results_df[eaug_col].idxmax()
    best_row = results_df.loc[best_idx]
    config = best_row.get("params", {})

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        st.metric("AUC", f"{best_row[auc_col]:.3f}")
    with col2:
        st.metric("Emisje CO₂ [kg]", f"{best_row[emissions_col]:.6f}")
    with col3:
        st.metric("EAUG (ΔAUC / kg CO₂)", f"{best_row[eaug_col]:.2e}")
    with col4:
        st.markdown(
            """
            **🌿 Zielona ocena:**  
            Im wyższy EAUG, tym **lepsza jakość przy niższych emisjach**.
            """
        )

    st.markdown("**Najważniejsze hiperparametry (RandomForest):**")
    if isinstance(config, dict):
        config_df = pd.DataFrame(
            {"Hiperparametr": list(config.keys()), "Wartość": list(config.values())}
        )
        st.table(config_df)
    else:
        st.write(config)

    st.markdown("---")
    st.subheader("📊 TOP konfiguracje wg EAUG")

    top_n = min(10, len(results_df))
    top_df = results_df.sort_values(by=eaug_col, ascending=False).head(top_n).copy()

    for c in [auc_col, emissions_col, eaug_col]:
        if c in top_df.columns:
            top_df[c] = top_df[c].astype(float).round(6)

    st.dataframe(top_df, use_container_width=True)

    st.markdown("---")
    st.subheader("🌍 Wykres AUC vs emisje CO₂")

    plot_df = results_df.copy()
    plot_df["params_str"] = plot_df["params"].astype(str)

    tooltip_cols = [auc_col, emissions_col, eaug_col, "params_str"]
    if "data_fraction" in plot_df.columns:
        tooltip_cols.append("data_fraction")

    chart = (
        alt.Chart(plot_df)
        .mark_circle(size=100, opacity=0.8)
        .encode(
            x=alt.X(auc_col, title="AUC (ROC)"),
            y=alt.Y(emissions_col, title="Emisje CO₂ [kg]"),
            color=alt.Color(eaug_col, title="EAUG"),
            tooltip=tooltip_cols,
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

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
        lines.append(f"Rodzina modelu (UI): {model_family}")
        lines.append(f"Maksymalna liczba trenowań (łącznie): {max_fits}")
        lines.append(f"Minimalne akceptowane AUC: {min_auc:.3f}")
        lines.append(f"Testowane frakcje danych: {', '.join(str(f) for f in selected_fractions)}\n")

        lines.append("== Najlepsza konfiguracja ==\n")
        lines.append(f"AUC (ROC): {best_row[auc_col]:.4f}")
        lines.append(f"Emisje CO₂ [kg]: {best_row[emissions_col]:.6f}")
        lines.append(f"EAUG (ΔAUC / kg CO₂): {best_row[eaug_col]:.2e}")
        if "data_fraction" in best_row:
            lines.append(f"Frakcja danych: {best_row['data_fraction']:.2f}")
        lines.append("")

        lines.append("Hiperparametry RandomForest:")
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
        lines.append(
            "W przyszłości raport może zostać automatycznie wygenerowany w formacie PDF "
            "i załączony do dokumentacji projektu."
        )

        return "\n".join(lines)

    generate_report = st.button("🧾 Generate PDF / text report")

    if generate_report:
        report_text = build_text_report(
            dataset=dataset_name,
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
    st.info(
        "Skonfiguruj parametry w panelu po lewej, ewentualnie wgraj własny plik CSV "
        "i kliknij **🚀 Run eco search**."
    )
