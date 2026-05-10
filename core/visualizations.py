"""
visualizations.py

Genera visualizaciones del análisis de fouling y optimización CIP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_fouling_evolution(
    flux_lmh: float,
    sdi_values: list,
    temperature_c: float,
    tds_mg_L: float,
    max_days: int = 200
) -> None:
    """
    Gráfica de evolución del NDP ratio para distintos valores de SDI.
    """
    from core.fouling_model import normalized_pressure_drop

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#2DC653", "#FFD166", "#F4A261", "#EF233C", "#6B0F1A"]

    for i, sdi in enumerate(sdi_values):
        days = list(range(0, max_days + 1, 5))
        ndp  = [normalized_pressure_drop(d, flux_lmh, sdi, tds_mg_L, temperature_c)
                for d in days]
        ax.plot(days, ndp, linewidth=2.5, color=colors[i], label=f"SDI = {sdi}")

    ax.axhline(1.15, color="red", linestyle="--", linewidth=1.5, label="CIP threshold (15%)")
    ax.axhline(1.0,  color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Days since last CIP", fontsize=11)
    ax.set_ylabel("Normalized Differential Pressure (NDP ratio)", fontsize=11)
    ax.set_title(f"Membrane Fouling Evolution\nFlux={flux_lmh} LMH | T={temperature_c}°C | TDS={tds_mg_L/1000:.0f} g/L",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, 1.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fouling_evolution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Guardado: fouling_evolution.png")


def plot_cip_frequency_heatmap(
    flux_values: list,
    sdi_values: list,
    temperature_c: float,
    tds_mg_L: float
) -> None:
    """
    Heatmap de frecuencia de CIP en función de flux y SDI.
    """
    from core.fouling_model import days_to_cip_threshold

    matrix = np.zeros((len(sdi_values), len(flux_values)))

    for i, sdi in enumerate(sdi_values):
        for j, flux in enumerate(flux_values):
            days = days_to_cip_threshold(flux, sdi, tds_mg_L, temperature_c)
            matrix[i, j] = round(365 / days, 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cip_risk", ["#2DC653", "#FFD166", "#EF233C"]
    )

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=1, vmax=15)

    ax.set_xticks(range(len(flux_values)))
    ax.set_xticklabels([f"{f:.0f}" for f in flux_values], fontsize=9)
    ax.set_yticks(range(len(sdi_values)))
    ax.set_yticklabels([f"{s:.1f}" for s in sdi_values], fontsize=9)
    ax.set_xlabel("Flux (LMH)", fontsize=11)
    ax.set_ylabel("SDI", fontsize=11)
    ax.set_title(f"CIP Frequency (times/year)\nT={temperature_c}°C | TDS={tds_mg_L/1000:.0f} g/L",
                 fontsize=13, fontweight="bold")

    for i in range(len(sdi_values)):
        for j in range(len(flux_values)):
            val = matrix[i, j]
            color = "white" if val > 8 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="CIP cleanings per year")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cip_frequency_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Guardado: cip_frequency_heatmap.png")


def plot_sdi_impact(
    flux_lmh: float,
    temperature_c: float,
    tds_mg_L: float
) -> None:
    """
    Gráfica del impacto del SDI en la frecuencia de CIP.
    El hallazgo principal del proyecto.
    """
    from core.fouling_model import days_to_cip_threshold

    sdi_range = np.arange(1.5, 5.1, 0.1)
    cip_per_year = []

    for sdi in sdi_range:
        days = days_to_cip_threshold(flux_lmh, sdi, tds_mg_L, temperature_c)
        cip_per_year.append(365 / days)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(sdi_range, cip_per_year, alpha=0.2, color="#E63946")
    ax.plot(sdi_range, cip_per_year, color="#E63946", linewidth=2.5)

    # Líneas de referencia SDI típicos
    for sdi_ref, label in [(2.0, "Excellent\npretreament"), (3.0, "Good"), (4.0, "Acceptable"), (5.0, "Limit")]:
        days = days_to_cip_threshold(flux_lmh, sdi_ref, tds_mg_L, temperature_c)
        freq = 365 / days
        ax.axvline(sdi_ref, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.annotate(f"SDI={sdi_ref}\n{freq:.1f}x/yr",
                    xy=(sdi_ref, freq),
                    xytext=(sdi_ref + 0.1, freq + 0.3),
                    fontsize=8, color="gray")

    ax.set_xlabel("SDI (post-pretreatment)", fontsize=11)
    ax.set_ylabel("CIP cleanings per year", fontsize=11)
    ax.set_title(f"Impact of Pretreatment Quality on CIP Frequency\nFlux={flux_lmh} LMH | T={temperature_c}°C",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sdi_impact.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Guardado: sdi_impact.png")