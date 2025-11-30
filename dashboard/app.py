import os
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# =========================================
# ŚCIEŻKI I IMPORT ECO_SEARCH Z robaczki_BHL/src
# =========================================
ROOT = Path(__file__).resolve().parent.parent  # robaczki_BHL
SRC_PATH = ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from eco_search import eco_random_search, _build_rf_pipeline  # noqa: F401, importujemy z eco_search


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
    Wgraj swój zbiór danych, wskaż kolumnę celu i rodzinę modelu.
    EcoSearch:
    
    * przeszuka przestrzeń konfiguracji,
    * wybierze model o najwyższym **EAUG (jakość / emisje CO₂)**,
    * wytrenuje go na pełnych danych,
    * pozwoli Ci wykorzystać ten model do predykcji.
    """
)


# =========================================
# PARAM_DISTRIBUTIONS – zależne od "rodziny" modelu (UI)
# (pod spodem wciąż RandomForest, ale o różnej złożoności)
# =========================================
def get_param_distributions(model_family: str):
    if model_family == "log_reg":
        # "lżejsze" RF – mała złożoność (symuluje prostszy model)
        return {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, 8],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
        }
    elif model_family == "mlp":
        # "cięższy" RF – bardziej złożony (symuluje bardziej skomplikowany model)
        return {
            "n_estimators": [200, 300, 400],
            "max_depth": [10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
    else:  # "random_forest"
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }


# =========================================
# SIDEBAR – USTAWIENIA ECOSEARCH
# =========================================
with st.sidebar:
    st.header("⚙️ Ustawienia EcoSearch")

    model_family = st.selectbox(
        "Rodzina modelu (konceptualnie)",
        options=["log_reg", "random_forest", "mlp"],
        format_func=lambda x: {
            "log_reg": "Logistic Regression (lekki)",
            "random_forest": "Random Forest",
            "mlp": "MLP (cięższy)",
        }.get(x, x),
    )

    max_fits = st.slider(
        "Maksymalna liczba trenowań modeli (łącznie)",
        min_value=5,
        max_value=200,
        value=40,
        step=5,
    )

    min_auc = st.slider(
        "Minimalnie akceptowalne AUC",
        min_value=0.50,
        max_value=0.99,
        value=0.75,
        step=0.01,
    )

    st.markdown("**Frakcje danych** wykorzystywane przez EcoSearch:")

    small_fractions = [0.01, 0.02, 0.03, 0.05, 0.08]
    medium_fractions = [0.1, 0.15, 0.2, 0.25, 0.33]
    large_fractions = [0.5, 0.75, 1.0]

    fractions_options = small_fractions + medium_fractions + large_fractions

    selected_fractions = st.multiselect(
        "Wybierz frakcje danych",
        options=fractions_options,
        default=[0.05, 0.1, 0.25, 0.5],
    )

    if not selected_fractions:
        st.warning("Wybierz chociaż jedną frakcję danych – ustawiam domyślnie 0.1.")
        selected_fractions = [0.1]

    selected_fractions = sorted(selected_fractions)
    max_data_fraction = max(selected_fractions)
    st.write(f"**Maks. frakcja danych:** {max_data_fraction:.2f}")

    run_button = st.button("🚀 Uruchom EcoSearch", type="primary")


# =========================================
# GŁÓWNA CZĘŚĆ – WŁASNY PLIK CSV
# =========================================
st.markdown("---")
st.subheader("📁 Własny zbiór danych (CSV)")

uploaded_file = st.file_uploader(
    "Przeciągnij tutaj plik CSV lub kliknij, aby wybrać",
    type=["csv"],
)

custom_target_col = None

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Nie udało się wczytać pliku: {e}")
        df_raw = None

    if df_raw is not None:
        st.write("Podgląd danych (pierwsze 5 wierszy):")
        st.dataframe(df_raw.head(), use_container_width=True)

        all_cols = list(df_raw.columns)
        if all_cols:
            custom_target_col = st.selectbox(
                "Wybierz kolumnę celu (y)",
                options=all_cols,
            )
            if custom_target_col:
                st.session_state["custom_data"] = {
                    "df": df_raw,
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
                "Sprawdź plik CSV."
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
                    df_frac["model_family"] = model_family
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
                    "max_fits": max_fits,
                    "min_auc": min_auc,
                    "selected_fractions": selected_fractions,
                    "X_df": X_df,
                    "y_ser": y_ser,
                    "model_family": model_family,
                }
                # czyścimy stary model, jeśli był
                if "final_model" in st.session_state:
                    del st.session_state["final_model"]


# =========================================
# PREZENTACJA WYNIKÓW I TRENING KOŃCOWEGO MODELU
# =========================================
if "results" in st.session_state:
    data_state = st.session_state["results"]
    results_df = data_state["results_df"]
    dataset_name = data_state["dataset"]
    max_fits = data_state["max_fits"]
    min_auc = data_state["min_auc"]
    selected_fractions = data_state["selected_fractions"]
    X_df_full = data_state["X_df"]
    y_ser_full = data_state["y_ser"]
    model_family = data_state["model_family"]

    auc_col = "roc_auc"
    emissions_col = "emissions_kg"
    eaug_col = "EAUG"

    st.markdown("---")
    st.subheader("🥇 Najlepsza znaleziona konfiguracja (wg EAUG)")

    best_idx = results_df[eaug_col].idxmax()
    best_row = results_df.loc[best_idx]
    config = best_row.get("params", {})

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        st.metric("AUC (na frakcji)", f"{best_row[auc_col]:.3f}")
    with col2:
        st.metric("Emisje CO₂ [kg]", f"{best_row[emissions_col]:.4f}")
    with col3:
        st.metric("EAUG (ΔAUC / kg CO₂)", f"{best_row[eaug_col]:.2e}")
    with col4:
        st.markdown(
            f"""
            **🌿 Zielona ocena**  
            Wybrana rodzina (UI): `{model_family}`.  
            Im wyższy EAUG, tym **lepsza jakość przy niższych emisjach**.
            """
        )

    st.markdown("**Wybrane hiperparametry (RandomForest):**")
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

    tooltip_cols = [auc_col, emissions_col, eaug_col, "params_str", "model_family"]
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

    # =========================================
    # TRENING KOŃCOWEGO MODELU NA PEŁNYCH DANYCH
    # =========================================
    st.markdown("---")
    st.subheader("🧠 Trening ostatecznego modelu na pełnych danych")

    X_full = X_df_full.values
    y_full = y_ser_full.values
    feature_cols = list(X_df_full.columns)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_full,
            y_full,
            test_size=0.25,
            random_state=42,
            stratify=y_full if len(np.unique(y_full)) > 1 else None,
        )

        rf_params = dict(config) if isinstance(config, dict) else {}
        rf_params = dict(rf_params)

        final_model = _build_rf_pipeline(rf_params, random_state=42)
        final_model.fit(X_train, y_train)

        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        final_auc = roc_auc_score(y_test, y_pred_proba)

        st.metric("AUC na zbiorze testowym (pełne dane)", f"{final_auc:.3f}")

        # wykres: przewidywanie vs faktyczny y
        plot_pred_df = pd.DataFrame(
            {
                "y_true": y_test,
                "y_pred_proba": y_pred_proba,
            }
        )
        plot_pred_df["y_true_jitter"] = plot_pred_df["y_true"] + np.random.normal(
            0, 0.02, size=len(plot_pred_df)
        )

        st.markdown("**Wykres: przewidywane prawdopodobieństwo vs faktyczny y (na teście)**")

        chart_pred = (
            alt.Chart(plot_pred_df)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("y_pred_proba", title="Przewidywane prawdopodobieństwo klasy pozytywnej"),
                y=alt.Y(
                    "y_true_jitter",
                    title="Rzeczywista klasa (z jitterem)",
                    scale=alt.Scale(domain=[-0.2, 1.2]),
                ),
                color=alt.Color("y_true:N", title="Rzeczywista klasa"),
                tooltip=["y_true", "y_pred_proba"],
            )
            .interactive()
        )
        st.altair_chart(chart_pred, use_container_width=True)

        # zapis modelu
        models_dir = ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "best_model_ecosearch.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)

        st.success(f"Model zapisany do pliku: `{model_path}`")

        # zapisujemy też do session_state, żeby użyć w interfejsie predykcji
        st.session_state["final_model"] = final_model
        st.session_state["final_model_features"] = feature_cols

    except Exception as e:
        st.error(f"Nie udało się wytrenować i zapisać ostatecznego modelu: {e}")


    # =========================================
    # INTERFEJS DO PREDYKCJI – WYKORZYSTANIE MODELU
    # =========================================
    if "final_model" in st.session_state:
        st.markdown("---")
        st.subheader("🔮 Wykorzystaj wytrenowany model do predykcji")

        final_model = st.session_state["final_model"]
        feature_cols = st.session_state["final_model_features"]

        tab_single, tab_batch = st.tabs(["Pojedyncza obserwacja", "Plik z nowymi danymi"])

        with tab_single:
            st.markdown("Wprowadź wartości cech dla jednej obserwacji:")

            col_values = {}
            for col in feature_cols:
                col_min = float(X_df_full[col].min())
                col_max = float(X_df_full[col].max())
                default = float(X_df_full[col].median())

                col_values[col] = st.number_input(
                    f"{col}",
                    value=default,
                    min_value=col_min,
                    max_value=col_max,
                )

            if st.button("🔍 Oblicz predykcję dla pojedynczej obserwacji"):
                X_new = pd.DataFrame([col_values], columns=feature_cols)
                proba = final_model.predict_proba(X_new.values)[:, 1][0]
                pred_class = int(proba >= 0.5)

                st.write(f"**Prawdopodobieństwo klasy pozytywnej:** `{proba:.4f}`")
                st.write(f"**Przewidywana klasa (threshold=0.5):** `{pred_class}`")

        with tab_batch:
            st.markdown(
                "Wgraj nowy plik CSV z danymi do predykcji. "
                "Kolumny zostaną dopasowane do cech użytych przy trenowaniu modelu."
            )

            pred_file = st.file_uploader(
                "Przeciągnij tutaj CSV z nowymi danymi",
                type=["csv"],
                key="pred_file_uploader",
            )

            if pred_file is not None:
                try:
                    df_new_raw = pd.read_csv(pred_file)
                except Exception as e:
                    st.error(f"Nie udało się wczytać pliku: {e}")
                    df_new_raw = None

                if df_new_raw is not None:
                    st.write("Podgląd nowych danych (pierwsze 5 wierszy):")
                    st.dataframe(df_new_raw.head(), use_container_width=True)

                    X_new = df_new_raw.select_dtypes(include=[np.number])
                    present_cols = [c for c in feature_cols if c in X_new.columns]
                    missing_cols = [c for c in feature_cols if c not in X_new.columns]

                    if len(present_cols) == 0:
                        st.error(
                            "W nowym pliku nie ma żadnych kolumn odpowiadających "
                            "cechom użytym przy trenowaniu modelu."
                        )
                    else:
                        if missing_cols:
                            st.warning(
                                "W nowym pliku brakuje części cech. "
                                "Brakujące cechy zostaną zastąpione średnimi ze zbioru treningowego. "
                                f"Brakujące cechy: {', '.join(missing_cols)}"
                            )
                        X_new_full = pd.DataFrame()
                        for c in feature_cols:
                            if c in X_new.columns:
                                X_new_full[c] = X_new[c]
                            else:
                                X_new_full[c] = X_df_full[c].mean()

                        proba_all = final_model.predict_proba(X_new_full.values)[:, 1]
                        pred_class_all = (proba_all >= 0.5).astype(int)

                        out_df = df_new_raw.copy()
                        out_df["pred_proba"] = proba_all
                        out_df["pred_class"] = pred_class_all

                        st.write("Podgląd predykcji (pierwsze 10 wierszy):")
                        st.dataframe(out_df.head(10), use_container_width=True)

                        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇️ Pobierz plik z predykcjami (CSV)",
                            data=csv_bytes,
                            file_name="predictions_ecosearch.csv",
                            mime="text/csv",
                        )

else:
    st.info(
        "Wgraj plik CSV, wybierz kolumnę celu, ustaw rodzinę modelu i parametry po lewej, "
        "a następnie kliknij **🚀 Uruchom EcoSearch**."
    )
