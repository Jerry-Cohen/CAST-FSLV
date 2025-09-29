
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


P_DURATION   = Path(r"FSLV Tracker - Duration of Protocols.csv")
P_MILESTONE  = Path(r"FSLV Tracker - Milestone Data (adjusted).csv")
P_PAIRS      = Path(r"PAIRS_2025_09_24_PTAP_approval_and_first_applied_to_patient_dates.xlsx - Pivot of pivot.csv")

ID_RE   = re.compile(r"^[A-Za-z]{2}\d{5}$")     # LLDDDDD
ISO_RE  = re.compile(r"^\d{4}-\d{2}-\d{2}$")    # yyyy-mm-dd
DMY_RE  = re.compile(r"^\d{2}-\d{2}-\d{4}$")    # dd-mm-yyyy

def norm_id(x: str) -> str:
    s = "" if x is None else str(x).strip().upper()
    return s if ID_RE.match(s) else ""

def parse_dates_safely(s: pd.Series) -> pd.Series:

    s = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    iso = s.str.match(ISO_RE)
    out.loc[iso] = pd.to_datetime(s.loc[iso], format="%Y-%m-%d", errors="coerce")

    dmy = (~iso) & s.str.match(DMY_RE)
    out.loc[dmy] = pd.to_datetime(s.loc[dmy], errors="coerce", dayfirst=True)

    rest = (~iso) & (~dmy)
    if rest.any():
        out.loc[rest] = pd.to_datetime(s.loc[rest], errors="coerce", dayfirst=True)  # handles 18-Aug-2025, etc.
    return out

def linear_fit_with_ci(x, y, x_pred):

    n = len(x)
    if n < 3:
        return None, None
    xbar, ybar = x.mean(), y.mean()
    Sxx = np.sum((x - xbar)**2)
    if Sxx == 0:
        return None, None
    Sxy = np.sum((x - xbar)*(y - ybar))
    b = Sxy / Sxx
    a = ybar - b * xbar
    yhat = a + b * x
    s2 = np.sum((y - yhat)**2) / (n - 2)
    s = float(np.sqrt(s2))
    tcrit = 1.96 if n > 30 else 2.13
    y_pred = a + b * x_pred
    ci_half = tcrit * np.sqrt(s2 * (1.0/n + (x_pred - xbar)**2 / Sxx))
    return (a, b, s), (y_pred, y_pred - ci_half, y_pred + ci_half)


def fit_two_sided_70_band(df):

    d = df[df["Delta_days"] >= 0].dropna(subset=["Calculated_PTA", "Actual_PTA"]).copy()
    if len(d) < 3:
        return None
    x = d["Calculated_PTA"].map(lambda dt: dt.toordinal()).astype(float).to_numpy()
    y = d["Actual_PTA"].map(lambda dt: dt.toordinal()).astype(float).to_numpy()

    n = x.size
    xbar = x.mean()
    ybar = y.mean()
    Sxx  = ((x - xbar)**2).sum()
    if Sxx == 0:
        return None
    Sxy  = ((x - xbar)*(y - ybar)).sum()
    b    = Sxy / Sxx
    a    = ybar - b*xbar
    yhat = a + b*x
    s    = float(np.sqrt(((y - yhat)**2).sum() / (n - 2)))


    tcrit = 1.036433389


    print(f"[Δ>=0 fit] n={n}, a={a:.6f}, b={b:.6f}, s={s:.6f}, t70_2sided≈z0.85={tcrit:.4f}")
    print("L_mean(x) = (a + b*x) - t70*s*sqrt(1/n + (x - xbar)^2/Sxx)")
    print(f"         = ({a:.6f}) + ({b:.6f})*x - {tcrit:.4f}*{s:.6f}*sqrt(1/{n} + (x - {xbar:.6f})^2/{Sxx:.6f})")
    print("U_mean(x) = (a + b*x) + t70*s*sqrt(1/n + (x - xbar)^2/Sxx)")
    print(f"         = ({a:.6f}) + ({b:.6f})*x + {tcrit:.4f}*{s:.6f}*sqrt(1/{n} + (x - {xbar:.6f})^2/{Sxx:.6f})")


    half_width_at_xbar = tcrit * s * math.sqrt(1.0/n)
    L_xbar = (a + b*xbar) - half_width_at_xbar
    U_xbar = (a + b*xbar) + half_width_at_xbar
    print(f"At x = xbar ({xbar:.6f}): L = {L_xbar:.6f}, U = {U_xbar:.6f} (half-width = {half_width_at_xbar:.6f})")

    return dict(a=a, b=b, s=s, tcrit=tcrit, xbar=xbar, Sxx=Sxx, n=n)


def mean_70_lo(x_pred, p):

    return (p["a"] + p["b"]*x_pred) - p["tcrit"]*p["s"]*np.sqrt(1.0/p["n"] + (x_pred - p["xbar"])**2 / p["Sxx"])

def mean_70_hi(x_pred, p):

    return (p["a"] + p["b"]*x_pred) + p["tcrit"]*p["s"]*np.sqrt(1.0/p["n"] + (x_pred - p["xbar"])**2 / p["Sxx"])


duration  = pd.read_csv(P_DURATION, dtype=str, encoding="utf-8-sig")
milestone = pd.read_csv(P_MILESTONE, dtype=str, encoding="utf-8-sig")
pairs     = pd.read_csv(P_PAIRS,     dtype=str, encoding="utf-8-sig")


if "Final List" not in duration.columns:
    raise KeyError("Expected 'Final List' in Duration.")
master = (
    duration["Final List"].map(norm_id)
    .replace("", np.nan).dropna().drop_duplicates().tolist()
)


weeks_col = "Weeks" if "Weeks" in duration.columns else next(
    (c for c in duration.columns if c.lower().startswith("weeks")), None
)
if weeks_col is None:
    raise KeyError("Couldn't find a 'Weeks' column in Duration.")
dur = duration[["Final List", weeks_col]].rename(columns={"Final List": "study_number", weeks_col: "Weeks"}).copy()
dur["study_number"] = dur["study_number"].map(norm_id)
dur = dur[dur["study_number"].isin(master)].drop_duplicates("study_number")
dur["Weeks"] = pd.to_numeric(dur["Weeks"], errors="coerce")


mil = milestone.rename(columns={milestone.columns[0]: "study_number"}).copy()
mil["study_number"] = mil["study_number"].map(norm_id)
mil = mil.iloc[1:].reset_index(drop=True)
PLANNED_COL = "First Study Subject Enrolled"
ACTUAL_COL  = "Unnamed: 4"
keep = ["study_number"]
if PLANNED_COL in mil.columns: keep.append(PLANNED_COL)
if ACTUAL_COL  in mil.columns: keep.append(ACTUAL_COL)
mil = mil[keep].copy()
if PLANNED_COL in mil.columns:
    mil.rename(columns={PLANNED_COL: "FSSE (Planned)"}, inplace=True)
    mil["FSSE (Planned)"] = parse_dates_safely(mil["FSSE (Planned)"])
if ACTUAL_COL in mil.columns:
    mil.rename(columns={ACTUAL_COL: "FSSE (Actual)"}, inplace=True)
    mil["FSSE (Actual)"]  = parse_dates_safely(mil["FSSE (Actual)"])
mil = mil[mil["study_number"].isin(master)].drop_duplicates("study_number")


pairs = pairs.rename(columns={pairs.columns[0]: "study_number"}).copy()
pairs["study_number"] = pairs["study_number"].map(norm_id)
if pairs.shape[1] < 2:
    raise KeyError("PAIRS CSV must have at least two columns (id + actual PTA).")
PAIRS_ACTUAL_PTA_COL = pairs.columns[1]
pairs = pairs[["study_number", PAIRS_ACTUAL_PTA_COL]].rename(columns={PAIRS_ACTUAL_PTA_COL: "Actual PTA"})
pairs["Actual PTA"] = parse_dates_safely(pairs["Actual PTA"])
pairs = pairs[pairs["study_number"].isin(master)].drop_duplicates("study_number")


merged = (
    pd.DataFrame({"study_number": master})
      .merge(dur, on="study_number", how="left")
      .merge(mil, on="study_number", how="left")
      .merge(pairs, on="study_number", how="left")
      .drop_duplicates("study_number")
      .reset_index(drop=True)
)

days = pd.to_numeric(merged["Weeks"], errors="coerce") * 7.0
td   = pd.to_timedelta(days.round(), unit="D")
if "FSSE (Planned)" in merged.columns:
    m = merged["FSSE (Planned)"].notna() & days.notna()
    merged.loc[:, "Calculated PTA (Planned)"] = pd.NaT
    merged.loc[m, "Calculated PTA (Planned)"] = merged.loc[m, "FSSE (Planned)"] + td[m]
if "FSSE (Actual)" in merged.columns:
    m = merged["FSSE (Actual)"].notna() & days.notna()
    merged.loc[:, "Calculated PTA (Actual)"] = pd.NaT
    merged.loc[m, "Calculated PTA (Actual)"] = merged.loc[m, "FSSE (Actual)"] + td[m]


rows = []
for _, r in merged.iterrows():
    actual = r.get("Actual PTA")
    if pd.notna(actual) and pd.notna(r.get("Calculated PTA (Planned)")):
        rows.append({
            "study_number": r["study_number"], "type": "Planned",
            "Calculated_PTA": r["Calculated PTA (Planned)"], "Actual_PTA": actual,
            "Delta_days": (actual - r["Calculated PTA (Planned)"]).days
        })
    if pd.notna(actual) and pd.notna(r.get("Calculated PTA (Actual)")):
        rows.append({
            "study_number": r["study_number"], "type": "Actual",
            "Calculated_PTA": r["Calculated PTA (Actual)"], "Actual_PTA": actual,
            "Delta_days": (actual - r["Calculated PTA (Actual)"]).days
        })
tidy = pd.DataFrame(rows)


print("--- Diagnostics ---")
print("Merged rows (should equal number of master IDs):", len(merged))
print("Weeks non-null:", merged["Weeks"].notna().sum())
print("FSSE Planned non-null:", merged.get("FSSE (Planned)", pd.Series(dtype='datetime64[ns]')).notna().sum())
print("FSSE Actual  non-null:", merged.get("FSSE (Actual)",  pd.Series(dtype='datetime64[ns]')).notna().sum())
print("Actual PTA non-null:", merged.get("Actual PTA",       pd.Series(dtype='datetime64[ns]')).notna().sum())
print("Tidy rows:", len(tidy))


def scatter_fit(df, title, band_params=None):

    if df.empty:
        print(f"{title}: no data"); return

    x = df["Calculated_PTA"].map(lambda d: d.toordinal()).astype(float).to_numpy()
    y = df["Actual_PTA"].map(lambda d: d.toordinal()).astype(float).to_numpy()
    if len(x) < 3:
        print(f"{title}: not enough points for fit"); return


    xbar, ybar = x.mean(), y.mean()
    Sxx = ((x - xbar)**2).sum()
    if Sxx == 0:
        print(f"{title}: vertical x"); return
    Sxy = ((x - xbar)*(y - ybar)).sum()
    b = Sxy / Sxx
    a = ybar - b*xbar
    yhat = a + b*x
    rmse = float(np.sqrt(((y - yhat)**2).sum() / (len(x) - 2)))

    x_pred = np.linspace(x.min(), x.max(), 200)
    y_pred = a + b*x_pred

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="studies")
    ax.plot(x_pred, y_pred, label=f"fit (RMSE≈{rmse:.0f} d)")

    if band_params is not None:
        L = mean_70_lo(x_pred, band_params)
        U = mean_70_hi(x_pred, band_params)
        ax.plot(x_pred, L, linestyle="-.", label="mean CI 70% (lower, Δ≥0 fit)")
        ax.plot(x_pred, U, linestyle="-.", label="mean CI 70% (upper, Δ≥0 fit)")


    mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([mn, mx], [mn, mx], linestyle="--", label="y = x")


    ticks = np.linspace(mn, mx, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([pd.Timestamp.fromordinal(int(v)).strftime("%Y-%m-%d") for v in ticks], rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels([pd.Timestamp.fromordinal(int(v)).strftime("%Y-%m-%d") for v in ticks])

    ax.set_title(title)
    ax.set_xlabel("Calculated PTA (date)")
    ax.set_ylabel("Actual PTA (date)")
    ax.legend()
    plt.tight_layout()
    plt.show()

def residual_hist(df, title):
    if df.empty:
        print(f"{title}: no data"); return
    vals = df["Delta_days"].dropna().values
    if len(vals) == 0:
        print(f"{title}: no residuals"); return
    fig, ax = plt.subplots()
    ax.hist(vals, bins=min(30, max(6, int(len(vals)**0.5))))
    ax.set_title(title + " — Residuals (Actual − Calculated) [days]")
    ax.set_xlabel("Δ days")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.show()

def bland_altman(df, title):
    if df.empty:
        print(f"{title}: no data"); return
    calc = df["Calculated_PTA"].astype("datetime64[ns]")
    act  = df["Actual_PTA"].astype("datetime64[ns]")
    x = ((calc + (act - calc)/2)).map(lambda d: d.toordinal()).values.astype(float)  # mean date
    y = (act - calc).dt.days.values.astype(float)                                    # Δ days
    if len(x) == 0:
        print(f"{title}: no points"); return
    m, sd = np.mean(y), (np.std(y, ddof=1) if len(y)>1 else np.nan)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.axhline(m, linestyle="-", label=f"mean Δ = {m:.1f} d")
    if not np.isnan(sd):
        ax.axhline(m + 1.96*sd, linestyle="--", label="±1.96 SD")
        ax.axhline(m - 1.96*sd, linestyle="--")
    ticks = np.linspace(x.min(), x.max(), 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([pd.Timestamp.fromordinal(int(v)).strftime("%Y-%m-%d") for v in ticks], rotation=45, ha="right")
    ax.set_title(title + " — Bland–Altman")
    ax.set_xlabel("Mean date (Calculated & Actual)")
    ax.set_ylabel("Δ days (Actual − Calculated)")
    ax.legend()
    plt.tight_layout()
    plt.show()

tidy_planned = tidy[tidy["type"] == "Planned"].copy()
tidy_actual  = tidy[tidy["type"] == "Actual"].copy()

band_planned = fit_two_sided_70_band(tidy_planned)
band_actual  = fit_two_sided_70_band(tidy_actual)

scatter_fit(tidy_planned, "Planned: Actual vs Calculated PTA", band_params=band_planned)
scatter_fit(tidy_actual,  "Actual:  Actual vs Calculated PTA", band_params=band_actual)

residual_hist(tidy_planned, "Planned")
residual_hist(tidy_actual,  "Actual")
bland_altman(tidy_planned, "Planned")
bland_altman(tidy_actual,  "Actual")
