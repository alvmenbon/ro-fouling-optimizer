"""
data_generator.py

Genera dataset sintético de escenarios de fouling en membranas RO.

Para cada combinación de condiciones de operación, simula
el tiempo hasta alcanzar el umbral de limpieza CIP usando
el motor físico de fouling.

El dataset resultante es el input para el modelo ML.
"""

import numpy as np
import pandas as pd
from core.fouling_model import days_to_cip_threshold


def generate_fouling_dataset(
    n_samples: int = 2000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Genera dataset sintético de escenarios de fouling.

    Variables de entrada (features):
        - flux_lmh:      flux de membrana (L/m²/h)
        - sdi:           Silt Density Index
        - tds_mg_L:      TDS del agua de alimentación (mg/L)
        - temperature_c: temperatura del agua (°C)

    Variable objetivo (target):
        - days_to_cip:   días hasta limpieza CIP

    Variable derivada:
        - cip_per_year:  limpiezas por año (365 / days_to_cip)

    Rangos basados en operación real de plantas SWRO:
        - flux:        10–18 LMH
        - SDI:         1.5–5.0 (post-pretratamiento)
        - TDS:         32000–45000 mg/L (agua de mar)
        - temperatura: 10–35°C
    """
    rng = np.random.default_rng(random_seed)

    flux        = rng.uniform(10.0, 18.0, n_samples)
    sdi         = rng.uniform(1.5,  5.0,  n_samples)
    tds         = rng.uniform(32000, 45000, n_samples)
    temperature = rng.uniform(10.0, 35.0, n_samples)

    days_to_cip = []

    for i in range(n_samples):
        days = days_to_cip_threshold(
            flux_lmh=flux[i],
            sdi=sdi[i],
            tds_mg_L=tds[i],
            temperature_c=temperature[i],
            threshold=1.15,
            max_days=365
        )
        days_to_cip.append(days)

        if (i + 1) % 500 == 0:
            print(f"  Generados {i+1}/{n_samples} escenarios...")

    df = pd.DataFrame({
        "flux_lmh":      flux,
        "sdi":           sdi,
        "tds_mg_L":      tds,
        "temperature_c": temperature,
        "days_to_cip":   days_to_cip,
    })

    df["cip_per_year"] = (365 / df["days_to_cip"]).round(2)

    return df


def dataset_summary(df: pd.DataFrame) -> None:
    """
    Imprime un resumen estadístico del dataset generado.
    """
    print(f"Dataset shape: {df.shape}")
    print(f"\nTarget variable (days_to_cip):")
    print(f"  Min:    {df['days_to_cip'].min()} días")
    print(f"  Max:    {df['days_to_cip'].max()} días")
    print(f"  Mean:   {df['days_to_cip'].mean():.1f} días")
    print(f"  Median: {df['days_to_cip'].median():.1f} días")
    print(f"\nCIP frequency distribution:")
    bins = [0, 2, 3, 4, 6, 12, 100]
    labels = ["<2/año", "2-3/año", "3-4/año", "4-6/año", "6-12/año", ">12/año"]
    df["cip_category"] = pd.cut(df["cip_per_year"], bins=bins, labels=labels)
    print(df["cip_category"].value_counts().sort_index())