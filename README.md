# ICEEMDAN–VMD Deep Learning Framework for AQI Forecasting

Source code, results, and figures behind the manuscript
**"A hybrid ICEEMDAN–VMD Deep Learning framework for Forecasting of AQI in Major
Indian cities."** Every table and data figure in the paper is reproducible from
the files in this repository.

The proposed model decomposes each city's AQI series with a hybrid **ICEEMDAN +
VMD** scheme (ICEEMDAN, then VMD refinement of the dominant high-frequency IMFs),
then forecasts each mode with deep-learning models and sums the mode forecasts.

---

## Repository layout

```
notebooks/
  decomposition_copy__main_training.ipynb   # MAIN: decomposition + trains all DL models,
                                            # writes results/<city>/h1/{metrics,forecasts,logs}
  significance_tests.ipynb                  # Wilcoxon + Diebold-Mariano tests
  paper_figures.py                          # regenerates the paper figures

results/<City>/h1/                          # one folder per city (Amravati, Bengaluru, Delhi, Mumbai)
  metrics/    <Model>.csv (20 rows = 20 runs) + All_Model_mean_Accuracy.csv
  forecasts/  per-model 20-run forecasts (input to significance tests + figures)
  logs/, figures/                           # training logs, SHAP / feature-importance plots

significance/                               # Wilcoxon / DM tables + significance plots
paper_figures/                              # fig0..fig8 PNGs + Fig1 framework diagram (SVG)
```

---

## How to run

### 1. Environment

Python 3.10–3.12. Install the dependencies:

```bash
pip install numpy pandas matplotlib scipy scikit-learn \
            torch xgboost lightgbm catboost \
            vmdpy EMD-signal shap joblib openpyxl
```

> `EMD-signal` provides the `PyEMD` package (CEEMDAN/EMD); `vmdpy` provides `VMD`.
> A CUDA-enabled `torch` speeds up training but CPU works.

### 2. Get the input data (required for retraining only)

The raw AQI series are **not** shipped in this repo. Place one Excel file per
city, each with a column named `AQI`, in a `data/` folder:

```
data/AQI_Amravati_2021_2025.xlsx
data/AQI_Bengaluru_2021_2025.xlsx
data/AQI_Delhi_2021_2025.xlsx
data/AQI_Mumbai_2021_2025.xlsx
```

### 3. Reproduce the model results (Tables in the paper)

Open `notebooks/decomposition_copy__main_training.ipynb` and, at the top of the
notebook, set the two hard-coded locations for the city you want:

- `file_path`  → the Excel file for that city (e.g. `.../AQI_Delhi_2021_2025.xlsx`)
- the output prefix (the `"Bengaluru/..."` strings) → the same city name

Key settings already in the notebook: `lag = 30`, `horizon = 1` (→ folder `h1`),
20 runs per model. Run all cells. It writes, per city, into
`results/<City>/h1/metrics|forecasts|logs`. The paper's per-city tables are the
**mean over the 20 runs** in each `metrics/<Model>.csv`, e.g.:

```python
import pandas as pd
print(round(pd.read_csv('results/Bengaluru/h1/metrics/GRU.csv')['RMSE'].mean(), 3))  # 1.633
```

Repeat for all four cities (this repo already contains the completed outputs, so
you can skip straight to steps 4–5 to regenerate figures/stats without retraining).

### 4. Regenerate the figures

`paper_figures.py` reads `results/<City>/h1/` and writes PNGs to `paper_figures/`:

```bash
python notebooks/paper_figures.py
```

Note: the script has a `BASE_DIR` path near the top — point it at this folder
before running.

### 5. Regenerate the significance tests

Open `notebooks/significance_tests.ipynb` (also uses a `BASE_DIR` at the top),
run all cells. It produces the Wilcoxon and Diebold–Mariano tables plus the
significance plots in `significance/`.

---

## Manuscript ↔ files (verified)

| Paper table | City | e.g. LSTM RMSE |
|---|---|---|
| Table 1 | Amravati | 2.749 |
| Table 3 | Bengaluru | 2.108 |
| Table 5 | Delhi | 4.351 |
| Table 7 | Mumbai | 1.401 |

| Paper figure | Content | File |
|---|---|---|
| Fig 1 | Proposed framework diagram | `paper_figures/Fig1_framework_diagram.svg` |
| Fig 2 | Temporal AQI, 4 cities | `paper_figures/fig0_all_data.png` |
| Fig 3 | Mean RMSE comparison | `paper_figures/fig8_crosscity_rmse.png` |
| Fig 4 | Actual vs predicted (best model) | `paper_figures/fig1_forecast_comparison.png` |
| Fig 5 | Scatter actual vs predicted | `paper_figures/fig2_scatter_plots.png` |
| Fig 6 | Box plot | `paper_figures/fig4_boxplot_stability.png` |

Models reported: LSTM, GRU, Bi-LSTM, CNN, CNN-LSTM, Transformer, Bi-GRU,
Deep LSTM, Deep GRU, CNN-GRU, Residual LSTM, Attention-Based LSTM, TCN,
Attention-Based CNN (`CNN_Attention`).
