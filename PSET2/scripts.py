import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# for unit conversions and periodogram 
from astropy import units as unt 
from astropy.timeseries import LombScargle 

# for fitting linear models 
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.model_selection import KFold

# for interactive and 3D plots 
from ipywidgets import interact
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42) # for reproducibility








#---Problem 3---#

from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii

data = ascii.read('data.dat', names = ['time', 'i-band_magnitude', 'uncertainty'])
data['time'] = data['time'] - 2450000
data = data[data['time']<7600]

#plt.plot(data['time'], data['i-band_magnitude'], marker='o', linestyle='none')
plt.errorbar(data['time'], data['i-band_magnitude'], yerr=data['uncertainty'], color = 'black', fmt='none', markersize=1, capsize=2, label='Data with error bars')
plt.gca().invert_yaxis()
plt.xlabel('Time [HJD - 2450000]')
plt.ylabel('i-band Magnitude')
plt.title('Anomalous Cepheid Data')
plt.legend()




def phase_plot(time, period, mag, yerr):

    time = np.mod(time/period, 2)

    plt.errorbar(time, mag, yerr, color = 'black', fmt='none', linewidth = 1,capsize=1, label='Data with error bars')
    plt.gca().invert_yaxis()
    plt.xlabel('Phase')
    plt.ylabel('i-band Magnitude')
    plt.title('Anomalous Cepheid Light Curve')
    plt.legend()
    plt.tight_layout()






phase_plot(time = data['time'], period = 0.833719, mag = data['i-band_magnitude'], yerr = data['uncertainty'])

# --- Trigonometric Linear Regression helpers (Problem 5) ---

def trig_design_matrix(time, period, k_max):
    """Return design matrix with columns [sin(2πkt/P), cos(2πkt/P)] for k=1..k_max."""
    return np.column_stack([
        f(2 * np.pi * k * time / period)
        for k in range(1, k_max + 1)
        for f in (np.sin, np.cos)
    ])


def fit_trig_linear(time, mag, yerr, period, k_max):
    """Weighted linear regression using trig basis; weights = 1/sigma^2.
    Intercept corresponds to y0 (mean magnitude).
    """
    X = trig_design_matrix(time, period, k_max)
    w = 1.0 / (np.asarray(yerr) ** 2)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, mag, sample_weight=w)
    return reg


def predict_trig(reg, time, period, k_max):
    """Predict magnitudes for given times using a fitted model and same k_max."""
    X = trig_design_matrix(time, period, k_max)
    return reg.predict(X)


def plot_trig_fit(time, mag, yerr, period, k_max, title_suffix=""):
    """Plot phased data with weighted trig-regression fit overlaid."""
    phase = np.mod(time / period, 2)
    order = np.argsort(phase)

    reg = fit_trig_linear(time, mag, yerr, period, k_max)
    y_fit = predict_trig(reg, time, period, k_max)

    plt.figure(figsize=(7, 4))
    plt.errorbar(phase[order], mag[order], yerr[order], fmt='none', color='black', linewidth=1, capsize=2, label='Data')
    plt.plot(phase[order], y_fit[order], linewidth=2, label=f'Fit (k_max={k_max})')
    plt.gca().invert_yaxis()
    plt.xlabel('Phase')
    plt.ylabel('i-band Magnitude')
    plt.title(f'Anomalous Cepheid Trig Fit {title_suffix}'.strip())
    plt.legend()
    plt.tight_layout()


# Quick static plot for k_max = 8 as requested
plot_trig_fit(
    time=data['time'].data,
    mag=data['i-band_magnitude'].data,
    yerr=data['uncertainty'].data,
    period=0.833719,
    k_max=8,
    title_suffix='(k_max=8)'
)

# Interactive widget to adjust k_max
@interact(k_max=(1, 20, 1))
def _update_trig_fit(k_max=8):
    plt.clf()
    plot_trig_fit(
        time=data['time'].data,
        mag=data['i-band_magnitude'].data,
        yerr=data['uncertainty'].data,
        period=0.833719,
        k_max=k_max,
        title_suffix=f'(k_max={k_max})'
    )

# --- Problem 6: Penalized Regression (Ridge / L2) ---
# Large k_max interactive sweep over log10(alpha) and CV-based alpha selection.

period = 0.833719

# Convenience alias to the existing design-matrix builder
trig_X = trig_design_matrix


def plot_ridge_trig(time, mag, yerr, period, k_max, log_alpha=-2.0):
    """Draw phased data and a Ridge-regularized trig fit for a given log10(alpha)."""
    alpha = 10.0**log_alpha
    X = trig_X(time, period, k_max)
    w = 1.0 / (np.asarray(yerr) ** 2)

    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(X, mag, sample_weight=w)
    y_fit = reg.predict(X)

    phase = np.mod(time / period, 2.0)
    order = np.argsort(phase)

    plt.figure(figsize=(7, 4))
    plt.errorbar(phase[order], mag[order], yerr[order], fmt='none', color='black', linewidth=1, capsize=2, label='Data')
    plt.plot(phase[order], y_fit[order], linewidth=2, label=fr'Ridge fit ($k_{{max}}={k_max}$, $\alpha={alpha:.2e}$)')
    plt.gca().invert_yaxis()
    plt.xlabel('Phase')
    plt.ylabel('i-band Magnitude')
    plt.title('Trigonometric Regression with L2 Penalty')
    plt.legend()
    plt.tight_layout()

    wmse = np.average((mag - y_fit) ** 2, weights=w)
    print(f"log10(alpha) = {log_alpha:.2f}, alpha = {alpha:.3e}, weighted MSE = {wmse:.6f}")


# Interactive control: explore log10(alpha) in [-6, 1] with a large harmonic basis
@interact(log_alpha=(-6.0, 1.0, 0.1))
def _ridge_alpha_sweep(log_alpha=-2.0):
    plt.clf()
    plot_ridge_trig(
        time=data['time'].data,
        mag=data['i-band_magnitude'].data,
        yerr=data['uncertainty'].data,
        period=period,
        k_max=64,
        log_alpha=log_alpha,
    )


# Cross-validation curve to pick alpha with a smaller k_max = 12
X12 = trig_X(data['time'].data, period, k_max=12)
Y   = data['i-band_magnitude'].data
W   = 1.0 / (data['uncertainty'].data ** 2)

alphas = np.logspace(-6, 1, 60)
cv_mse = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for a in alphas:
    fold_mse = []
    for tr, va in kf.split(X12):
        r = Ridge(alpha=a, fit_intercept=True)
        r.fit(X12[tr], Y[tr], sample_weight=W[tr])
        yhat = r.predict(X12[va])
        fold_mse.append(np.average((Y[va] - yhat) ** 2, weights=W[va]))
    cv_mse.append(np.mean(fold_mse))

best_idx = int(np.argmin(cv_mse))
best_alpha = alphas[best_idx]
print(f"[CV] Best alpha: {best_alpha:.3e}")

plt.figure(figsize=(6, 4))
plt.semilogx(alphas, cv_mse, '-o', markersize=3)
plt.axvline(best_alpha, color='r', linestyle='--', label='Best α')
plt.xlabel('alpha')
plt.ylabel('Weighted CV MSE')
plt.title('Ridge CV curve (k_max=12)')
plt.legend()
plt.tight_layout()

# Plot phased fit using the CV-selected alpha for k_max=12
r_best = Ridge(alpha=best_alpha, fit_intercept=True)
r_best.fit(X12, Y, sample_weight=W)
yfit12 = r_best.predict(X12)
phase12 = np.mod(data['time'].data / period, 2.0)
ord12 = np.argsort(phase12)

plt.figure(figsize=(7, 4))
plt.errorbar(phase12[ord12], Y[ord12], data['uncertainty'].data[ord12], fmt='none', color='black', linewidth=1, capsize=2, label='Data')
plt.plot(phase12[ord12], yfit12[ord12], linewidth=2, label=fr'RidgeCV fit ($k_{{max}}=12$, $\alpha={best_alpha:.2e}$)')
plt.gca().invert_yaxis()
plt.xlabel('Phase')
plt.ylabel('i-band Magnitude')
plt.title('CV-selected Ridge penalty (k_max=12)')
plt.legend()
plt.tight_layout()

# --- Exercise 6 Extra Credit: model vs time with dense evaluation + zoom ---
# Build dense grids for smooth model rendering
_time_all = data['time'].data
_mag_all  = data['i-band_magnitude'].data
_err_all  = data['uncertainty'].data

# Full range dense grid
_time_dense_full = np.linspace(_time_all.min(), _time_all.max(), 5000)
_X_dense_full = trig_X(_time_dense_full, period, k_max=12)
_yfit_dense_full = r_best.predict(_X_dense_full)

# Zoomed dense grid over [5500, 5514]
_t0, _t1 = 5500.0, 5514.0
_time_dense_zoom = np.linspace(_t0, _t1, 2000)
_X_dense_zoom = trig_X(_time_dense_zoom, period, k_max=12)
_yfit_dense_zoom = r_best.predict(_X_dense_zoom)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharey=True)

# Top: full time span
axes[0].errorbar(_time_all, _mag_all, yerr=_err_all, fmt='none', color='black', linewidth=1, capsize=2, label='Observed Data')
axes[0].plot(_time_dense_full, _yfit_dense_full, color='red', linewidth=2, alpha=0.8, label=fr'RidgeCV fit ($k_{{max}}=12$, $\alpha={best_alpha:.2e}$)')
axes[0].invert_yaxis()
axes[0].set_xlabel('Time [HJD - 2450000]')
axes[0].set_ylabel('i-band Magnitude')
axes[0].set_title('Cepheid Light Curve with RidgeCV Fit (time domain)')
axes[0].legend()

# Bottom: zoom on [5500, 5514]
axes[1].errorbar(_time_all, _mag_all, yerr=_err_all, fmt='none', color='black', linewidth=1, capsize=2, label='Observed Data')
axes[1].plot(_time_dense_zoom, _yfit_dense_zoom, color='red', linewidth=2, alpha=0.8, label='RidgeCV fit (dense)')
axes[1].invert_yaxis()
axes[1].set_xlabel('Time [HJD - 2450000]')
axes[1].set_ylabel('i-band Magnitude')
axes[1].set_title('Zoom: Pulsations over two weeks')
axes[1].set_xlim(_t0, _t1)
axes[1].legend()

plt.tight_layout()

# --- Problem 7: Periodogram of residuals ---
# Subtract CV-selected model (k_max=12, best_alpha) from data and compute Lomb–Scargle.

residuals = Y - yfit12  # residuals at observed times (use same weighting as fit if needed)

freq_res, power_res = LombScargle(
    data['time'].data,
    residuals,
    data['uncertainty'].data
).autopower(nyquist_factor=3000)

period_res = 1.0 / freq_res

plt.figure(figsize=(9, 4.8))
plt.plot(period_res, power_res, color='black', linewidth=1, label='Residuals LS power')

# Mark the fundamental period and a few harmonics for reference
P0 = period
plt.axvline(P0, color='crimson', linewidth=3, alpha=0.3, label=fr'Period = {P0:.2f} days')
for k in range(2, 9):
    lbl = 'Harmonics' if k == 2 else None
    plt.axvline(P0 / k, color='skyblue', linewidth=3, alpha=0.3, label=lbl)

plt.xlabel('Period [days]')
plt.ylabel('Power')
plt.title('Lomb–Scargle Periodogram of Residuals')
plt.ylim(bottom=0)
plt.xlim(0, 2)  # same axis range as before
plt.legend()
plt.tight_layout()

#---Problem 4---#

frequency, power = LombScargle(data['time'], data['i-band_magnitude'], data['uncertainty']).autopower(nyquist_factor=3000) # I increased the nyquist_factor until I saw period values for near 0 days

plt.plot(1/frequency, power, color = 'black', linewidth=1)

plt.axvline(0.833719, color='red', linewidth = 3, alpha = 0.3, label='Known Period = 0.833719 days')
for k in range(2, 9):
    lbl = '1/k Harmonics' if k == 2 else None
    plt.axvline(0.833719 / k, color='blue', linewidth=3, alpha=0.3, label=lbl)

plt.xlabel('Period [days]')
plt.ylabel('Lomb-Scargle Power')
plt.ylim(bottom = 0)
plt.title('Lomb-Scargle Periodogram')
plt.legend()
plt.xlim(0,2)
plt.tight_layout()


# --- Extra Credit 1: OGLE catalog & class for penalized trig fits ---
# Utilities to (a) read the OGLE Cepheid catalog you exported (data.txt)
# and (b) fit penalized trigonometric regression to multiple stars
# without duplicating code.

from pathlib import Path
import os

# Reuse the design builder from above
# (trig_design_matrix and trig_X already defined)


def load_ogle_catalog(path: str) -> pd.DataFrame:
    """Load an OGLE Cepheid summary table and normalize key columns.

    Expected format is a tab-separated file with leading comment lines
    beginning with '#'. Column names vary between exports; this function
    normalizes them and ensures a float `Period` column exists.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")

    cat = pd.read_csv(path, sep="\t", comment="#", header=0, engine="python")
    # Normalize column names
    cat.columns = [c.strip().replace(".", "_") for c in cat.columns]

    def _first_existing(cols: list[str]) -> str:
        for c in cols:
            if c in cat.columns:
                return c
        return ""

    # Determine period column (OGLE often uses 'P_1')
    pcol = _first_existing(["Period", "P_1", "P1", "P"])
    if not pcol:
        raise KeyError(f"Could not find a period column in catalog. Available: {list(cat.columns)}")
    # Create canonical Period column as float
    cat["Period"] = pd.to_numeric(cat[pcol], errors="coerce")

    # Determine an ID-like column
    idcol = _first_existing(["ID", "Id", "ID_OGLE_IV", "ID_OGLE_III", "ID_OGLE_II"]) or cat.columns[0]
    cat["ID"] = cat[idcol].astype(str).str.strip()

    return cat


class CepheidPenalized:
    """Penalized trigonometric regression for a single Cepheid.

    Parameters
    ----------
    period : float
        Period in days.
    k_max : int, default 12
        Number of harmonics to include (both sin and cos for k=1..k_max).
    penalty : {"ridge", "lasso"}
        Which L2/L1 penalty to use.
    alpha : float or None
        Regularization strength. If None, a 5-fold CV search is used over
        logspace(1e-6 .. 1e0).
    random_state : int
        Random state for KFold shuffling in CV.
    """

    def __init__(self, period: float, k_max: int = 12,
                 penalty: str = "ridge", alpha: Optional[float] = None,
                 random_state: int = 42):
        self.period = float(period)
        self.k_max = int(k_max)
        self.penalty = penalty.lower()
        self.alpha = alpha
        self.random_state = random_state
        self.model = None

    def _X(self, time):
        return trig_X(np.asarray(time), self.period, self.k_max)

    def fit(self, time, mag, yerr):
        X = self._X(time)
        y = np.asarray(mag)
        w = 1.0 / (np.asarray(yerr) ** 2)

        if self.alpha is None:
            # Cross-validated alpha selection
            alphas = np.logspace(-6, 0, 60)
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_mse = []
            for a in alphas:
                fold = []
                for tr, va in kf.split(X):
                    if self.penalty == "ridge":
                        m = Ridge(alpha=a, fit_intercept=True)
                    elif self.penalty == "lasso":
                        m = Lasso(alpha=a, fit_intercept=True, max_iter=5000)
                    else:
                        raise ValueError("penalty must be 'ridge' or 'lasso'")
                    m.fit(X[tr], y[tr], sample_weight=w[tr])
                    yhat = m.predict(X[va])
                    fold.append(np.average((y[va] - yhat) ** 2, weights=w[va]))
                cv_mse.append(np.mean(fold))
            self.alpha = float(alphas[int(np.argmin(cv_mse))])

        # Final refit on all data using chosen alpha
        if self.penalty == "ridge":
            self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        else:
            self.model = Lasso(alpha=self.alpha, fit_intercept=True, max_iter=10000)
        self.model.fit(X, y, sample_weight=w)
        return self

    def predict(self, time):
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model.predict(self._X(time))

    def phase(self, time):
        return np.mod(np.asarray(time) / self.period, 2.0)

    def plot_phased(self, time, mag, yerr, label_prefix: str = ""):
        phase = self.phase(time)
        order = np.argsort(phase)
        yfit = self.predict(time)
        plt.figure(figsize=(7, 4))
        plt.errorbar(phase[order], np.asarray(mag)[order], np.asarray(yerr)[order],
                     fmt='none', color='black', linewidth=1, capsize=2, label='Data')
        plt.plot(phase[order], yfit[order], linewidth=2,
                 label=f"{label_prefix} {self.penalty.title()} (k={self.k_max}, α={self.alpha:.2e})".strip())
        plt.gca().invert_yaxis()
        plt.xlabel('Phase')
        plt.ylabel('i-band Magnitude')
        plt.title('Cepheid phased light curve with penalized trig fit')
        plt.legend(); plt.tight_layout()


# --- Helper to read an OGLE light-curve file ---
from astropy.io import ascii as _ascii

def read_ogle_lightcurve(path: str):
    """Return (time, mag, yerr) from a typical OGLE *.dat lightcurve file.

    The function is intentionally tolerant of column names. It looks for
    time columns containing 'HJD' or 'time', magnitude columns containing
    'I' or 'mag', and error columns containing 'err' or 'sigma'.
    """
    t = _ascii.read(path, comment='#')
    # Try to guess column names
    cols = [c.lower() for c in t.colnames]
    def pick(preds):
        for p in preds:
            for i, c in enumerate(cols):
                if p in c:
                    return t.columns[t.colnames[i]].data
        return t.columns[t.colnames[0]].data
    time = pick(["hjd", "time"])
    mag  = pick([" i ".strip(), "mag"])  # look for 'I' or 'mag'
    yerr = pick(["err", "sigma"]) if any("err" in c or "sigma" in c for c in cols) else np.full_like(mag, np.nan, dtype=float)
    return np.asarray(time, dtype=float), np.asarray(mag, dtype=float), np.asarray(yerr, dtype=float)


# --- Batch runner: fit a few Cepheids from the catalog ---

def run_extra_credit(catalog_path: str = "data.txt", lc_dir: str = "lightcurves", 
                     n_stars: int = 3, k_max: int = 12, penalty: str = "ridge"):
    """Example driver that loads the catalog, picks `n_stars` with periods
    in a sensible range, searches for their lightcurve files in `lc_dir`,
    fits, and saves phased plots into `plots/`.

    Expects files named like '<ID>.dat' inside `lc_dir` (e.g., 'OGLE-LMC-CEP-0101.dat').
    """
    cat = load_ogle_catalog(catalog_path)
    # Pick plausible classical Cepheids: 0.5 < P < 20 days
    subset = cat[(cat["Period"] > 0.5) & (cat["Period"] < 20.0)].head(n_stars)

    outdir = Path("plots"); outdir.mkdir(parents=True, exist_ok=True)

    for _, row in subset.iterrows():
        star_id = str(row["ID"]) if "ID" in row.index else str(row.iloc[0])
        period = float(row["Period"]) if not pd.isna(row["Period"]) else np.nan
        fpath = Path(lc_dir) / f"{star_id}.dat"
        if not fpath.exists():
            print(f"[skip] no lightcurve file for {star_id} at {fpath}")
            continue
        time, mag, err = read_ogle_lightcurve(fpath)
        # If errors missing, fall back to unweighted (~equal weights)
        if np.all(np.isnan(err)):
            err = np.full_like(mag, np.nanmedian(np.abs(mag - np.nanmedian(mag))), dtype=float)
        model = CepheidPenalized(period=period, k_max=k_max, penalty=penalty, alpha=None)
        model.fit(time, mag, err)
        model.plot_phased(time, mag, err, label_prefix=star_id)
        # Save figure
        outfile = outdir / f"{star_id}_phased_{penalty}_k{k_max}.png"
        plt.savefig(outfile, dpi=180)
        plt.close()
        print(f"[ok] {star_id}: alpha={model.alpha:.3e}, saved {outfile}")

# Example (commented out). Uncomment to run once you place lightcurves:
# run_extra_credit(catalog_path="data.txt", lc_dir="lightcurves", n_stars=5, k_max=12, penalty="ridge")
# You can also try LASSO by penalty="lasso".