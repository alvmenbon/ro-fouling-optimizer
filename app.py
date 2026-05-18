"""
app.py

Streamlit interface for the RO Fouling Optimizer.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from core.fouling_model import days_to_cip_threshold, normalized_pressure_drop
from core.visualizations import plot_cip_frequency_heatmap

st.set_page_config(
    page_title="RO Fouling Optimizer",
    layout="wide"
)

st.title("RO Fouling Optimizer")
st.markdown("""
Optimize CIP cleaning frequency for RO membranes during the **EPC design phase**.
Based on a physics-based fouling model validated against SWRO operational ranges.
""")

# --- Sidebar ---
st.sidebar.header("Design Parameters")

flux = st.sidebar.slider(
    "Design flux (LMH)",
    min_value=10.0,
    max_value=18.0,
    value=12.0,
    step=0.5
)

sdi = st.sidebar.slider(
    "SDI target (post-pretreatment)",
    min_value=1.5,
    max_value=5.0,
    value=3.0,
    step=0.1
)

temperature = st.sidebar.slider(
    "Water temperature (°C)",
    min_value=10.0,
    max_value=35.0,
    value=20.0,
    step=0.5
)

tds = st.sidebar.number_input(
    "Feed TDS (mg/L)",
    min_value=30000,
    max_value=45000,
    value=38000,
    step=500
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
**RO Fouling Optimizer** is an open source tool for process engineers 
working on seawater desalination projects.

Built by **Álvaro Mendoza**, Process Engineer specialized 
in water treatment and desalination.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/alvaro-mendoza-bonilla)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/alvmenbon/ro-fouling-optimizer)
""")

# --- Results ---
days = days_to_cip_threshold(flux, sdi, tds, temperature)
cip_per_year = round(365 / days, 1)

col1, col2, col3 = st.columns(3)
col1.metric("Days to CIP", f"{days} days")
col2.metric("CIP frequency", f"{cip_per_year} times/year")
col3.metric("vs 3/year baseline", f"{round(cip_per_year - 3, 1):+.1f}")

st.markdown("---")

# --- Fouling evolution ---
st.subheader("Fouling Evolution")

days_range = list(range(0, 201, 5))
ndp_values = [normalized_pressure_drop(d, flux, sdi, tds, temperature) for d in days_range]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(days_range, ndp_values, color="#E63946", linewidth=2.5)
ax.axhline(1.15, color="red", linestyle="--", linewidth=1.5, label="CIP threshold (15%)")
ax.axvline(days, color="gray", linestyle=":", linewidth=1.5, label=f"CIP day ({days})")
ax.set_xlabel("Days since last CIP")
ax.set_ylabel("NDP ratio")
ax.set_title(f"Membrane Fouling — Flux={flux} LMH | SDI={sdi} | T={temperature}°C")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0.95, max(1.5, max(ndp_values) + 0.05))
st.pyplot(fig)

# --- SDI sensitivity ---
st.subheader("SDI Sensitivity Analysis")

sdi_range = np.arange(1.5, 5.1, 0.1)
cip_range = [round(365 / days_to_cip_threshold(flux, s, tds, temperature), 1) for s in sdi_range]

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.fill_between(sdi_range, cip_range, alpha=0.2, color="#E63946")
ax2.plot(sdi_range, cip_range, color="#E63946", linewidth=2.5)
ax2.axvline(sdi, color="gray", linestyle="--", linewidth=1.5, label=f"Current SDI={sdi}")
ax2.axhline(3, color="blue", linestyle=":", linewidth=1.5, label="Typical design criterion (3/year)")
ax2.set_xlabel("SDI (post-pretreatment)")
ax2.set_ylabel("CIP cleanings per year")
ax2.set_title(f"Impact of Pretreatment Quality | Flux={flux} LMH")
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

st.markdown("---")
st.caption("""
**Limitations**: Physics model based on power-law cake filtration (n=0.7). 
Validated against published SWRO operational ranges. 
Synthetic data only — recalibrate with real plant data when available.
""")