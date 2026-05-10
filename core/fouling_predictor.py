"""
fouling_predictor.py

Modelo ML para predecir días hasta limpieza CIP en membranas RO.

Entrena un Random Forest Regressor sobre el dataset sintético
generado por el motor físico de fouling.

Features:
    flux_lmh, sdi, tds_mg_L, temperature_c

Target:
    days_to_cip
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path



FEATURES = ["flux_lmh", "sdi", "tds_mg_L", "temperature_c"]
TARGET   = "days_to_cip"


def train_model(df: pd.DataFrame) -> tuple:
    """
    Entrena el modelo Random Forest sobre el dataset sintético.

    Retorna:
        model:   modelo entrenado
        scaler:  scaler ajustado
        metrics: dict con métricas de evaluación
    """
    X = df[FEATURES]
    y = df[TARGET]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Escalado
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Modelo
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_sc, y_train)

    # Evaluación
    y_pred = model.predict(X_test_sc)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(
        model, scaler.transform(X), y,
        cv=5, scoring="r2"
    )

    metrics = {
        "mae":     round(mae, 2),
        "r2":      round(r2, 4),
        "cv_r2_mean": round(cv_scores.mean(), 4),
        "cv_r2_std":  round(cv_scores.std(), 4),
        "n_train": len(X_train),
        "n_test":  len(X_test),
    }

    return model, scaler, metrics, X_test, y_test, y_pred


def feature_importance(model: RandomForestRegressor) -> pd.DataFrame:
    """
    Retorna la importancia de cada feature en el modelo.
    """
    importance = pd.DataFrame({
        "feature":    FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return importance


def plot_predictions(y_test, y_pred, metrics: dict) -> None:
    """
    Gráfica de predicciones vs valores reales.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: predicho vs real
    ax1.scatter(y_test, y_pred, alpha=0.4, color="#1D3557", s=15)
    lims = [0, max(y_test.max(), y_pred.max()) + 10]
    ax1.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax1.set_xlabel("Actual days to CIP")
    ax1.set_ylabel("Predicted days to CIP")
    ax1.set_title(f"Predicted vs Actual\nR²={metrics['r2']} | MAE={metrics['mae']} days")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residuals
    residuals = np.array(y_pred) - np.array(y_test)
    ax2.hist(residuals, bins=40, color="#E63946", alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Residual (predicted - actual) days")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Residual Distribution\nMean={residuals.mean():.1f} | Std={residuals.std():.1f}")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Guardado: outputs/model_evaluation.png")


def predict_cip(
    model: RandomForestRegressor,
    scaler: StandardScaler,
    flux_lmh: float,
    sdi: float,
    tds_mg_L: float,
    temperature_c: float
) -> dict:
    """
    Predice días hasta CIP para unas condiciones de operación dadas.

    Retorna dict con días predichos y frecuencia anual.
    """
    X = pd.DataFrame([{
        "flux_lmh":      flux_lmh,
        "sdi":           sdi,
        "tds_mg_L":      tds_mg_L,
        "temperature_c": temperature_c
    }])

    X_sc = scaler.transform(X)
    days = model.predict(X_sc)[0]

    return {
        "days_to_cip":  round(days),
        "cip_per_year": round(365 / days, 2),
        "cip_interval": f"Every {round(days)} days ({round(365/days, 1)} times/year)"
    }