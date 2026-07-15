"""
paper_figures.py
Generate publication-quality figures for the ICEEMDAN-VMD AQI forecasting paper.

Run: python paper_figures.py
Outputs saved to: simple_aqi/paper_figures/
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from scipy import stats

# ── Global plot style ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = r"D:\Projects\Python Scripts\AQI 2026\simple_aqi"
DATA_DIR = r"D:\Projects\Python Scripts\AQI 2026\final_data"
FIG_DIR  = os.path.join(BASE_DIR, "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Study constants ───────────────────────────────────────────────────────────
CITIES = {
    "Bengaluru": "AQI_Bengaluru_2021_2025.xlsx",
    "Delhi":     "AQI_Delhi_2021_2025.xlsx",
    "Mumbai":    "AQI_Mumbai_2021_2025.xlsx",
    "Amravati":  "AQI_Amravati_2021_2025.xlsx",
}
CITY_ORDER = ["Delhi", "Mumbai", "Bengaluru", "Amravati"]

BEST_MODELS = {
    "Delhi":     "Deep_GRU",
    "Mumbai":    "Deep_GRU",
    "Bengaluru": "Deep_GRU",
    "Amravati":  "CNN_LSTM",
}

DL_MODELS = [
    "LSTM", "GRU", "BiLSTM", "CNN", "CNN_LSTM", "Transformer",
    "BiGRU", "Deep_LSTM", "Deep_GRU", "CNN_GRU", "Residual_LSTM",
    "Attention_LSTM", "TCN", "CNN_Attention",
]
MODEL_LABELS = {
    "LSTM": "LSTM", "GRU": "GRU", "BiLSTM": "BiLSTM", "CNN": "CNN",
    "CNN_LSTM": "CNN-LSTM", "Transformer": "Transformer", "BiGRU": "BiGRU",
    "Deep_LSTM": "Deep-LSTM", "Deep_GRU": "Deep-GRU", "CNN_GRU": "CNN-GRU",
    "Residual_LSTM": "Res-LSTM", "Attention_LSTM": "Att-LSTM",
    "TCN": "TCN", "CNN_Attention": "CNN-Att",
}

LAG, HORIZON = 30, 1
METRICS = ["RMSE", "MAE", "SMAPE", "MASE", "R2"]

CITY_COLORS = {
    "Delhi":     "#762a83",
    "Mumbai":    "#2166ac",
    "Bengaluru": "#1a9641",
    "Amravati":  "#d6604d",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    print(f"  Saved -> {path}")


def _label(m: str) -> str:
    return MODEL_LABELS.get(m, m)


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_city_data() -> dict:
    """Load all forecast matrices, per-run metrics, and true values for every city."""
    city_data = {}
    for city, fname in CITIES.items():
        df      = pd.read_excel(os.path.join(DATA_DIR, fname))
        series  = df["AQI"].values.astype(float)
        dates   = pd.to_datetime(df["Date"])
        n       = len(series)
        tr      = int(0.6 * n)
        va      = int(0.2 * n)
        test    = series[tr + va:]
        n_ts    = len(test) - LAG - HORIZON
        y_true  = test[LAG + HORIZON - 1: LAG + HORIZON - 1 + n_ts]
        start   = tr + va + LAG + HORIZON - 1
        t_dates = dates.iloc[start: start + n_ts].reset_index(drop=True)

        fc_dir  = os.path.join(BASE_DIR, city, "results", "h1", "forecasts")
        met_dir = os.path.join(BASE_DIR, city, "results", "h1", "metrics")

        forecasts = {}
        for f in sorted(os.listdir(fc_dir)):
            if not f.endswith(".csv"):
                continue
            mat = pd.read_csv(os.path.join(fc_dir, f)).values  # (n_ts, 20)
            forecasts[f.replace(".csv", "")] = mat

        per_run = {}
        for f in sorted(os.listdir(met_dir)):
            if "_IMF_importance" in f or "All_Model" in f:
                continue
            if not f.endswith(".csv"):
                continue
            name = f.replace(".csv", "")
            mdf  = pd.read_csv(os.path.join(met_dir, f))
            if "RMSE" in mdf.columns:
                per_run[name] = mdf

        summary = pd.read_csv(os.path.join(met_dir, "All_Model_mean_Accuracy.csv"))
        city_data[city] = {
            "y_true": y_true, "dates": t_dates,
            "forecasts": forecasts, "per_run": per_run, "summary": summary,
        }
        print(f"  Loaded {city}: {n_ts} test samples, {len(forecasts)} forecast matrices")
    return city_data


# ─────────────────────────────────────────────────────────────────────────────
# Figure 0 — Full AQI time series with train / validation / test split (2x2)
# ─────────────────────────────────────────────────────────────────────────────

def fig0_all_data() -> None:
    """Full AQI series for all 4 cities with split regions and AQI category bands."""

    # Indian AQI category thresholds and colours
    AQI_BANDS = [
        (0,   50,  "#00e400", "Good"),
        (50,  100, "#92d050", "Satisfactory"),
        (100, 200, "#ffff00", "Moderate"),
        (200, 300, "#ff7e00", "Poor"),
        (300, 400, "#ff0000", "Very Poor"),
        (400, 500, "#7e0023", "Severe"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 8),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.25})
    axes = axes.flatten()

    for ax, city in zip(axes, CITY_ORDER):
        df      = pd.read_excel(os.path.join(DATA_DIR, CITIES[city]))
        df["Date"] = pd.to_datetime(df["Date"])
        series  = df["AQI"].values.astype(float)
        dates   = df["Date"]
        n       = len(series)
        tr      = int(0.6 * n)
        va      = int(0.2 * n)
        te      = n - tr - va

        d_tr_end = dates.iloc[tr - 1]
        d_va_end = dates.iloc[tr + va - 1]

        # fix ylim first so annotations land correctly
        ymax = max(series.max() * 1.12, 60)
        ax.set_ylim(0, ymax)
        ax.set_xlim(dates.iloc[0], dates.iloc[-1])

        # ── AQI category background bands ──────────────────────────────────
        for lo, hi, color, _ in AQI_BANDS:
            ax.axhspan(lo, hi, facecolor=color, alpha=0.08, zorder=0)

        # ── Split shading ──────────────────────────────────────────────────
        ax.axvspan(dates.iloc[0],  d_tr_end,       alpha=0.10, color="#2166ac", zorder=1)
        ax.axvspan(d_tr_end,       d_va_end,        alpha=0.10, color="#f4a582", zorder=1)
        ax.axvspan(d_va_end,       dates.iloc[-1],  alpha=0.10, color="#1a9641", zorder=1)

        # ── Split boundary lines ───────────────────────────────────────────
        ax.axvline(d_tr_end, color="#2166ac", lw=1.4, ls="--", zorder=2)
        ax.axvline(d_va_end, color="#d6604d", lw=1.4, ls="--", zorder=2)

        # ── AQI time series ────────────────────────────────────────────────
        ax.plot(dates, series, color=CITY_COLORS[city], lw=0.9, zorder=3)

        # ── Split labels — axes-fraction x, data y ─────────────────────────
        # x fractions for mid-point of each segment
        x_frac_tr = (tr / 2) / n
        x_frac_va = (tr + va / 2) / n
        x_frac_te = (tr + va + te / 2) / n
        y_top = ymax * 0.97

        # use blended transform: x in axes coords, y in data coords
        from matplotlib.transforms import blended_transform_factory
        btrans = blended_transform_factory(ax.transAxes, ax.transData)
        for xf, txt in [(x_frac_tr, "Train"),
                        (x_frac_va, "Val"),
                        (x_frac_te, "Test")]:
            ax.text(xf, y_top, txt, transform=btrans,
                    ha="center", va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, zorder=5))

        # ── City label — smart corner placement ───────────────────────────────
        idx_arr = np.arange(n)
        def _density(corner, bx, by):
            if   corner == 'bl': tm = idx_arr < n*bx;      vm = series < ymax*by
            elif corner == 'tl': tm = idx_arr < n*bx;      vm = series > ymax*(1-by)
            elif corner == 'br': tm = idx_arr > n*(1-bx);  vm = series < ymax*by
            else:                tm = idx_arr > n*(1-bx);  vm = series > ymax*(1-by)
            return float(np.sum(tm & vm)) / n

        CPOS = {
            'bl': (0.02, 0.03, 'left',  'bottom'),
            'tl': (0.02, 0.97, 'left',  'top'),
            'br': (0.98, 0.03, 'right', 'bottom'),
            'tr': (0.98, 0.97, 'right', 'top'),
        }
        c_rank = sorted(CPOS, key=lambda k: _density(k, 0.18, 0.16))
        cx, cy, cha, cva = CPOS[c_rank[0]]
        ax.text(cx, cy, city, transform=ax.transAxes,
                va=cva, ha=cha, fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.88, zorder=6))

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.set_ylabel("AQI")
        ax.set_xlabel("Date")

    # ── Shared legend ──────────────────────────────────────────────────────
    split_handles = [
        mpatches.Patch(color="#2166ac", alpha=0.45, label="Training set"),
        mpatches.Patch(color="#f4a582", alpha=0.65, label="Validation set"),
        mpatches.Patch(color="#1a9641", alpha=0.45, label="Test set"),
    ]
    fig.legend(handles=split_handles, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.85,
               bbox_to_anchor=(0.5, 0.0))

    fig.subplots_adjust(bottom=0.12)
    _save(fig, "fig0_all_data.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Actual vs Predicted time series (2x2 grid)
# ─────────────────────────────────────────────────────────────────────────────

def fig1_forecast_comparison(city_data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, city in zip(axes, CITY_ORDER):
        data  = city_data[city]
        best  = BEST_MODELS[city]
        y     = data["y_true"]
        dates = data["dates"]
        mat   = data["forecasts"].get(best)

        if mat is None:
            ax.text(0.5, 0.5, f"Forecast not found: {best}", transform=ax.transAxes, ha="center")
            continue

        n     = min(len(y), mat.shape[0])
        y_    = y[:n]
        pmean = mat[:n].mean(axis=1)
        pstd  = mat[:n].std(axis=1)
        x     = dates[:n]

        ax.fill_between(x, pmean - pstd, pmean + pstd,
                        alpha=0.20, color=CITY_COLORS[city], label=r"$\pm$1 Std Dev (20 runs)")
        ax.plot(x, y_,    color="#111111", lw=1.5, label="Actual AQI")
        ax.plot(x, pmean, color=CITY_COLORS[city], lw=1.5, ls="--",
                label=f"{_label(best)} (Predicted)")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        # Use official mean metrics from All_Model_mean_Accuracy.csv
        summary = data["summary"].set_index("Model")
        rmse_v  = float(summary.loc[best, "RMSE"])
        r2_v    = float(summary.loc[best, "R2"])
        ax.text(0.02, 0.97,
                f"RMSE = {rmse_v:.4f}   " + r"$R^2$" + f" = {r2_v:.4f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))

        # City label inside plot (top-right) instead of title
        ax.text(0.98, 0.97, city, transform=ax.transAxes,
                va="top", ha="right", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))

        ax.set_ylabel("AQI")
        ax.set_xlabel("Date")
        ax.legend(loc="upper center", fontsize=8, ncol=3,
                  bbox_to_anchor=(0.5, -0.18))

    plt.tight_layout()
    _save(fig, "fig1_forecast_comparison.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Scatter plots: Actual vs Predicted (2x2 grid)
# ─────────────────────────────────────────────────────────────────────────────

def fig2_scatter_plots(city_data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, city in zip(axes, CITY_ORDER):
        data  = city_data[city]
        best  = BEST_MODELS[city]
        y     = data["y_true"]
        mat   = data["forecasts"].get(best)

        if mat is None:
            continue

        n     = min(len(y), mat.shape[0])
        y_    = y[:n]
        pmean = mat[:n].mean(axis=1)

        ax.scatter(y_, pmean, alpha=0.55, s=14, color=CITY_COLORS[city],
                   edgecolors="none", label="Test samples")

        mn, mx = min(y_.min(), pmean.min()), max(y_.max(), pmean.max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, label="Perfect fit")

        slope, intercept, r, *_ = stats.linregress(y_, pmean)
        xl = np.linspace(mn, mx, 200)
        ax.plot(xl, slope * xl + intercept, "-", color=CITY_COLORS[city],
                lw=1.5, alpha=0.8, label="Regression line")

        # Use official mean metrics from All_Model_mean_Accuracy.csv
        summary = data["summary"].set_index("Model")
        rmse_v  = float(summary.loc[best, "RMSE"])
        r2_v    = float(summary.loc[best, "R2"])
        ax.text(0.97, 0.03,
                r"$R^2$" + f" = {r2_v:.4f}\nRMSE = {rmse_v:.4f}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.80))

        ax.text(0.50, 0.97, city, transform=ax.transAxes,
                va="center", ha="center", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))

        ax.set_xlabel("Actual AQI")
        ax.set_ylabel("Predicted AQI")
        ax.legend(loc="best", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    _save(fig, "fig2_scatter_plots.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — RMSE horizontal bar chart (2x2, one per city)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_rmse_bar_chart(city_data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, city in zip(axes, CITY_ORDER):
        summary = city_data[city]["summary"]
        dl_df   = summary[summary["Model"].isin(DL_MODELS)].copy()
        dl_df   = dl_df.sort_values("RMSE", ascending=False)
        best    = BEST_MODELS[city]

        models = dl_df["Model"].tolist()
        rmse   = dl_df["RMSE"].values
        colors = [CITY_COLORS[city] if m == best else "#c0c0c0" for m in models]

        ax.barh(range(len(models)), rmse, color=colors,
                edgecolor="white", linewidth=0.5, height=0.72)

        for i, val in enumerate(rmse):
            ax.text(val + max(rmse) * 0.01, i, f"{val:.3f}", va="center", fontsize=8)

        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([_label(m) for m in models], fontsize=9)
        ax.set_xlabel("RMSE (Mean over 20 runs)")
        ax.set_xlim(0, max(rmse) * 1.18)

        # City label inside plot
        ax.text(0.98, 0.02, city, transform=ax.transAxes,
                va="bottom", ha="right", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))

        best_patch  = mpatches.Patch(color=CITY_COLORS[city], label=f"Best: {_label(best)}")
        other_patch = mpatches.Patch(color="#c0c0c0", label="Other DL models")
        ax.legend(handles=[best_patch, other_patch], loc="lower right", fontsize=8)

    plt.tight_layout()
    _save(fig, "fig3_rmse_bar_chart.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Per-run RMSE boxplots (2x2, all DL models sorted by median)
# ─────────────────────────────────────────────────────────────────────────────

def fig4_boxplot_stability(city_data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    for ax, city in zip(axes, CITY_ORDER):
        data    = city_data[city]
        best    = BEST_MODELS[city]
        per_run = data["per_run"]

        dl_avail  = [m for m in DL_MODELS if m in per_run]
        dl_sorted = sorted(dl_avail, key=lambda m: np.median(per_run[m]["RMSE"].values))

        plot_data  = [per_run[m]["RMSE"].values for m in dl_sorted]
        labels     = [_label(m) for m in dl_sorted]
        box_colors = [CITY_COLORS[city] if m == best else "#c8c8c8" for m in dl_sorted]

        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True,
                        notch=False, vert=True, widths=0.6,
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))

        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.80)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.8)
        for whisker in bp["whiskers"]:
            whisker.set_color("#555555")
        for cap in bp["caps"]:
            cap.set_color("#555555")

        ax.set_ylabel("RMSE")
        ax.set_xlabel("Model (sorted by median RMSE, left = best)")
        ax.tick_params(axis="x", labelsize=8, rotation=40)

        ax.text(0.98, 0.97, city, transform=ax.transAxes,
                va="top", ha="right", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))

        best_patch  = mpatches.Patch(color=CITY_COLORS[city], alpha=0.8,
                                      label=f"Best: {_label(best)}")
        other_patch = mpatches.Patch(color="#c8c8c8", alpha=0.8, label="Other DL models")
        ax.legend(handles=[best_patch, other_patch], fontsize=8, loc="upper left")

    plt.tight_layout()
    _save(fig, "fig4_boxplot_stability.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Wilcoxon signed-rank p-value heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig5_wilcoxon_heatmap() -> None:
    sig_file = os.path.join(BASE_DIR, "Significance_Tests_All_Cities.xlsx")
    if not os.path.exists(sig_file):
        print("  [!] Significance_Tests_All_Cities.xlsx not found -- skipping Fig 5")
        return

    wl_df = pd.read_excel(sig_file, sheet_name="Wilcoxon_Test")
    wl_df = wl_df[wl_df["Comparison_Model"].isin(DL_MODELS)]

    pivot = wl_df.pivot_table(
        index="Comparison_Model", columns="City",
        values="p_value", aggfunc="first",
    ).astype(float)

    cols  = [c for c in CITY_ORDER if c in pivot.columns]
    pivot = pivot[cols]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    labels_y = [_label(m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=(9, max(5, len(pivot) * 0.55 + 1.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.10)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(labels_y, fontsize=9)

    def _star(p: float) -> str:
        if np.isnan(p):
            return "N/A"
        s = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        return f"{p:.3f}\n{s}"

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            txt_color = "white" if (not np.isnan(val) and val < 0.025) else "black"
            ax.text(j, i, _star(val), ha="center", va="center",
                    fontsize=7.5, color=txt_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("p-value", fontsize=10)

    ax.text(
        0.5, -0.07,
        "*** p < 0.001   ** p < 0.01   * p < 0.05   ns: not significant",
        transform=ax.transAxes, ha="center", fontsize=8.5, style="italic",
    )
    plt.tight_layout()
    _save(fig, "fig5_wilcoxon_heatmap.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Comprehensive performance metric heatmap (5 metrics x all models)
# ─────────────────────────────────────────────────────────────────────────────

def fig6_metric_heatmap(city_data: dict) -> None:
    metric_tables: dict[str, dict] = {m: {} for m in METRICS}

    for city in CITY_ORDER:
        summary = city_data[city]["summary"]
        dl_df   = summary[summary["Model"].isin(DL_MODELS)].set_index("Model")
        for metric in METRICS:
            if metric in dl_df.columns:
                metric_tables[metric][city] = dl_df[metric]

    dfs = {}
    for metric in METRICS:
        dfs[metric] = pd.DataFrame(metric_tables[metric]).reindex(DL_MODELS)[CITY_ORDER]

    fig, axes = plt.subplots(1, 5, figsize=(24, 9))

    for ax, metric in zip(axes, METRICS):
        vals = dfs[metric].values.astype(float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        norm = (vals - vmin) / (vmax - vmin + 1e-12)
        if metric == "R2":
            norm = 1 - norm

        cmap = "RdYlGn_r" if metric != "R2" else "RdYlGn"
        ax.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(CITY_ORDER)))
        ax.set_xticklabels(CITY_ORDER, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(DL_MODELS)))
        if ax is axes[0]:
            ax.set_yticklabels([_label(m) for m in DL_MODELS], fontsize=9)
        else:
            ax.set_yticklabels([])

        # Metric name as xlabel (replaces title)
        ax.set_xlabel(metric, fontsize=12, fontweight="bold", labelpad=8)

        def _fmt(v: float, met: str) -> str:
            if np.isnan(v):
                return "-"
            if met in ("RMSE", "MAE"):
                return f"{v:.3f}"
            return f"{v:.4f}"

        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v  = vals[i, j]
                nv = norm[i, j]
                txt_color = "white" if nv > 0.70 else "black"
                ax.text(j, i, _fmt(v, metric),
                        ha="center", va="center", fontsize=7, color=txt_color)

        # Black border around best cell per city column
        for j in range(len(CITY_ORDER)):
            col = vals[:, j]
            best_row = int(np.nanargmax(col) if metric == "R2" else np.nanargmin(col))
            ax.add_patch(matplotlib.patches.Rectangle(
                (j - 0.5, best_row - 0.5), 1, 1,
                linewidth=2, edgecolor="black", facecolor="none",
            ))

    plt.tight_layout()
    _save(fig, "fig6_metric_heatmap.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Residual distribution: histogram + normal fit (2x2)
# ─────────────────────────────────────────────────────────────────────────────

def fig7_error_distribution(city_data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, city in zip(axes, CITY_ORDER):
        data  = city_data[city]
        best  = BEST_MODELS[city]
        y     = data["y_true"]
        mat   = data["forecasts"].get(best)

        if mat is None:
            continue

        n     = min(len(y), mat.shape[0])
        y_    = y[:n]
        pmean = mat[:n].mean(axis=1)
        resid = pmean - y_

        ax.hist(resid, bins=28, color=CITY_COLORS[city], alpha=0.65,
                edgecolor="white", density=True, label="Residual density")

        mu, sigma = float(np.mean(resid)), float(np.std(resid))
        x_n = np.linspace(resid.min() - 0.5, resid.max() + 0.5, 300)
        ax.plot(x_n, stats.norm.pdf(x_n, mu, sigma),
                "k-", lw=1.8, label=r"N($\mu$=" + f"{mu:.3f}" + r", $\sigma$=" + f"{sigma:.3f})")

        ax.axvline(0,  color="red",  lw=1.3, ls="--", alpha=0.8, label="Zero error")
        ax.axvline(mu, color="navy", lw=1.1, ls=":",  alpha=0.8, label=f"Mean={mu:.3f}")

        sample = resid if len(resid) <= 5000 else np.random.choice(resid, 5000, replace=False)
        _, sw_p = stats.shapiro(sample)
        ax.text(0.97, 0.97,
                f"Shapiro-Wilk\np = {sw_p:.4f}",
                transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))

        ax.text(0.03, 0.97, f"{city} — {_label(best)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))

        ax.set_xlabel(r"Residual  (Predicted $-$ Actual)  [AQI]")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "fig7_error_distribution.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Cross-city RMSE: grouped bar chart (14 models x 4 cities)
# ─────────────────────────────────────────────────────────────────────────────

def fig8_crosscity_rmse(city_data: dict) -> None:
    rows = []
    for city in CITY_ORDER:
        summary = city_data[city]["summary"]
        dl_df   = summary[summary["Model"].isin(DL_MODELS)].set_index("Model")
        rows.append(dl_df["RMSE"])

    rmse_df  = pd.concat(rows, axis=1, keys=CITY_ORDER).reindex(DL_MODELS)
    rmse_df  = rmse_df.loc[rmse_df.mean(axis=1).sort_values().index]
    n_models = len(rmse_df)
    n_cities = len(CITY_ORDER)
    bar_w    = 0.18
    x        = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(20, 7))

    for ci, city in enumerate(CITY_ORDER):
        offset = (ci - n_cities / 2 + 0.5) * bar_w
        vals   = rmse_df[city].values
        ax.bar(x + offset, vals, bar_w,
               color=CITY_COLORS[city], label=city,
               edgecolor="white", linewidth=0.4, alpha=0.88)

        best = BEST_MODELS[city]
        if best in rmse_df.index:
            bi = rmse_df.index.get_loc(best)
            ax.bar(bi + offset, vals[bi], bar_w,
                   color=CITY_COLORS[city], edgecolor="black",
                   linewidth=1.8, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in rmse_df.index], rotation=35, ha="right", fontsize=9.5)
    ax.set_ylabel("RMSE (Mean over 20 runs)")
    ax.set_xlabel("Deep Learning Model  (sorted by mean RMSE across cities)")
    ax.legend(title="City", fontsize=9, title_fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save(fig, "fig8_crosscity_rmse.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading city data ...")
    city_data = load_city_data()
    print(f"\nGenerating figures -> {FIG_DIR}\n")

    print("Fig 0: Full AQI time series with splits ...")
    fig0_all_data()

    print("Fig 1: Actual vs Predicted time series ...")
    fig1_forecast_comparison(city_data)

    print("Fig 2: Scatter plots ...")
    fig2_scatter_plots(city_data)

    print("Fig 3: RMSE bar chart ...")
    fig3_rmse_bar_chart(city_data)

    print("Fig 4: Per-run RMSE boxplots ...")
    fig4_boxplot_stability(city_data)

    print("Fig 5: Wilcoxon significance heatmap ...")
    fig5_wilcoxon_heatmap()

    print("Fig 6: Comprehensive metric heatmap ...")
    fig6_metric_heatmap(city_data)

    print("Fig 7: Residual distribution ...")
    fig7_error_distribution(city_data)

    print("Fig 8: Cross-city RMSE grouped bar chart ...")
    fig8_crosscity_rmse(city_data)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
