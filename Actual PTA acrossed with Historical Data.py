

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats


MILESTONES_CSV = r"4_Analytical Method for FSLV Prediction - Milestone Data.csv"
APPROVALS_CSV  = r"PAIRS_2025_09_24_PTAP_approval_and_first_applied_to_patient_dates.xlsx - Pivot of pivot.csv"
OUT_PREFIX     = "approval_gap_model"

ISO_RX = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*$")

def to_dt_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    iso_mask = s.str.match(ISO_RX)
    out = pd.to_datetime(s.where(iso_mask), errors="coerce", dayfirst=False)
    rem = pd.to_datetime(s.where(~iso_mask), errors="coerce", dayfirst=True)
    out = out.fillna(rem).dt.normalize()
    return out

# ---------- parsing the two files exactly as uploaded ----------
def parse_milestones(path: str) -> pd.DataFrame:
    m = pd.read_csv(path, header=None, dtype=str)

    data = m.iloc[3:].reset_index(drop=True).copy()
    data.rename(columns={0: "study"}, inplace=True)
    data["study"] = data["study"].astype(str).str.strip()

    header_names = m.iloc[1].fillna("").astype(str)
    header_flags = m.iloc[2].fillna("").astype(str).str.upper()

    def find_pair(patterns):
        pat = re.compile(patterns, flags=re.I)
        c_plan = c_act = None
        for j, name in header_names.items():
            if pat.search(name) and header_flags.get(j, "") == "PLANNED_DATE":
                c_plan = j
                c_act = j + 1
                break
        if c_plan is None or header_flags.get(c_act, "") != "ACTUAL_DATE":
            raise KeyError(f"Could not find planned/actual pair for milestone: {patterns}")
        return c_plan, c_act

    fsse_p, fsse_a = find_pair(r"first\s*study\s*subject\s*enrolled")
    lsse_p, lsse_a = find_pair(r"last\s*study\s*subject\s*enrolled")
    lslv_p, lslv_a = find_pair(r"last\s*(study\s*)?subject\s*last\s*visit")

    out = pd.DataFrame({
        "study": data["study"],
        "FSSE_P": to_dt_series(data[fsse_p]),
        "FSSE_A": to_dt_series(data[fsse_a]),
        "LSSE_P": to_dt_series(data[lsse_p]),
        "LSSE_A": to_dt_series(data[lsse_a]),
        "LSLV_P": to_dt_series(data[lslv_p]),
        "LSLV_A": to_dt_series(data[lslv_a]),
    })
    out = out[out["study"].str.len() > 0].reset_index(drop=True)
    return out

def parse_approvals(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str)
    header_row = raw.index[raw[0].astype(str).str.strip().str.lower() == "protocol number"]
    if len(header_row) == 0:
        raise KeyError("Could not find 'Protocol Number' header row in approvals CSV.")
    h = header_row[0]
    headers = raw.iloc[h].astype(str).tolist()
    df = raw.iloc[h+1:].reset_index(drop=True).copy()
    df.columns = [str(x).strip() for x in headers]

    if "Protocol Number" not in df.columns:
        raise KeyError("Expected 'Protocol Number' column in approvals CSV.")
    if "Min of Date of final patient approval" not in df.columns:
        raise KeyError("Expected 'Min of Date of final patient approval' column in approvals CSV.")

    df = df.rename(columns={"Protocol Number": "study"})
    df["study"] = df["study"].astype(str).str.strip()
    df["APPROVAL_MIN"] = to_dt_series(df["Min of Date of final patient approval"])
    df = df.loc[df["study"].str.len() > 0, ["study", "APPROVAL_MIN"]].dropna(subset=["APPROVAL_MIN"]).reset_index(drop=True)
    return df

# ---------- audit & model helpers ----------
def audit_mode(milestones: pd.DataFrame, approvals: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "planned":
        FSSE, LSSE, LSLV = "FSSE_P", "LSSE_P", "LSLV_P"
    elif mode == "actual":
        FSSE, LSSE, LSLV = "FSSE_A", "LSSE_A", "LSLV_A"
    else:
        raise ValueError("mode must be 'planned' or 'actual'")

    a = milestones[["study", FSSE, LSSE, LSLV]].copy()
    a["missing_FSSE"] = a[FSSE].isna()
    a["missing_LSSE"] = a[LSSE].isna()
    a["missing_LSLV"] = a[LSLV].isna()
    a["missing_any_milestone"] = a[["missing_FSSE","missing_LSSE","missing_LSLV"]].any(axis=1)

    a = a.merge(approvals.rename(columns={"APPROVAL_MIN":"approval"}), on="study", how="left")
    a["missing_approval"] = a["approval"].isna()
    a["included"] = (~a["missing_any_milestone"]) & (~a["missing_approval"])

    def reason_row(r):
        reasons = []
        if r["missing_FSSE"]: reasons.append("FSSE missing")
        if r["missing_LSSE"]: reasons.append("LSSE missing")
        if r["missing_LSLV"]: reasons.append("LSLV missing")
        if r["missing_approval"]: reasons.append("approval missing")
        return ", ".join(reasons) if reasons else "included"
    a["reason"] = a.apply(reason_row, axis=1)
    return a

def build_inclusion_matrix(audit_planned: pd.DataFrame, audit_actual: pd.DataFrame) -> pd.DataFrame:
    p = audit_planned[["study","included","reason"]].rename(columns={"included":"planned_included","reason":"planned_reason"})
    q = audit_actual[["study","included","reason"]].rename(columns={"included":"actual_included","reason":"actual_reason"})
    m = p.merge(q, on="study", how="outer").fillna({"planned_included": False, "actual_included": False})
    # For readability, sort: both included first, then planned-only, actual-only, neither
    m["_sortkey"] = (m["planned_included"].astype(int) + m["actual_included"].astype(int))
    m = m.sort_values(["_sortkey","study"], ascending=[False, True]).drop(columns=["_sortkey"])
    return m

def build_mode_dataset(milestones: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "planned":
        FSSE, LSSE, LSLV = "FSSE_P", "LSSE_P", "LSLV_P"
    else:
        FSSE, LSSE, LSLV = "FSSE_A", "LSSE_A", "LSLV_A"

    df = milestones[["study", FSSE, LSSE, LSLV]].dropna(subset=[FSSE, LSSE, LSLV]).copy()
    if df.empty:
        return df.assign(A_days=np.nan, anchor=pd.NaT)

    anchor = df[FSSE].min()
    df["A_days"] = (df[LSLV] - df[LSSE]).dt.days + (df[FSSE] - anchor).dt.days
    df["anchor"] = anchor
    return df

def fit_constant_gap(joined: pd.DataFrame):
    anchor = joined["anchor"].iloc[0]
    A_abs = anchor + pd.to_timedelta(joined["A_days"], unit="D")
    gap_days = (joined["APPROVAL_MIN"] - A_abs).dt.days.dropna()

    n = int(gap_days.size)
    if n < 3:
        raise ValueError("Need at least 3 complete rows to estimate gap.")

    gap_hat = float(gap_days.mean())
    sd = float(gap_days.std(ddof=1))
    t_q = float(stats.t.ppf(0.70, df=n-1))      # one-sided 70%
    pred_sd = float(sd * np.sqrt(1 + 1/n))      # prediction sd

    def predict_from_A_days(a_days: float):
        pt = anchor + pd.to_timedelta(a_days + gap_hat, unit="D")
        lb = anchor + pd.to_timedelta(a_days + gap_hat - t_q * pred_sd, unit="D")
        return pt.normalize(), lb.normalize()

    return dict(n=n, gap_hat_days=gap_hat, gap_sd_days=sd,
                t_0_70=t_q, pred_sd_days=pred_sd, anchor=anchor,
                predict_from_A_days=predict_from_A_days)

# -------------------- MAIN --------------------
def run(milestones_path: str, approvals_path: str, out_prefix: str = OUT_PREFIX):
    milestones = parse_milestones(milestones_path)
    approvals  = parse_approvals(approvals_path)

    # Independent audits (no cross-mode reuse)
    audit_p = audit_mode(milestones, approvals, mode="planned")
    audit_a = audit_mode(milestones, approvals, mode="actual")

    # Inclusion matrix to demonstrate independence
    incl = build_inclusion_matrix(audit_p, audit_a)
    incl.to_csv("inclusion_matrix.csv", index=False)

    # Save audits
    audit_p.to_csv("audit_planned.csv", index=False)
    audit_a.to_csv("audit_actual.csv", index=False)

    # -------- Planned pipeline (independent) --------
    dP = build_mode_dataset(milestones, "planned")
    jP = dP.merge(approvals, on="study", how="inner").dropna(subset=["APPROVAL_MIN"]).reset_index(drop=True)

    print("\n==== PLANNED ====")
    if len(jP) >= 3:
        fitP = fit_constant_gap(jP)
        print(f"Studies used (n): {fitP['n']}")
        print(f"GAP_hat (days): {fitP['gap_hat_days']:.2f} | sd: {fitP['gap_sd_days']:.2f} | "
              f"t_0.70: {fitP['t_0_70']:.4f} | pred_sd: {fitP['pred_sd_days']:.2f}")
        print(f"Anchor date: {fitP['anchor'].date()}")
        print("Prediction equation:")
        print("  approval_hat = anchor + timedelta( A_days + GAP_hat )")
        print("  A_days = (LSLV − LSSE) + (FSSE − anchor)  [days]")
        print("  LB70 = anchor + timedelta( A_days + GAP_hat − t_{0.70,df} * sd * sqrt(1 + 1/n) )")

        preds = []
        for r in jP.itertuples(index=False):
            pt, lb = fitP["predict_from_A_days"](r.A_days)
            preds.append(dict(study=r.study, A_days=r.A_days,
                              pred_approval=str(pt.date()),
                              LB70_approval=str(lb.date()),
                              actual_approval=str(r.APPROVAL_MIN.date())))
        pd.DataFrame(preds).to_csv(f"{out_prefix}_planned.csv", index=False)
        print(f"Per-study predictions saved: {out_prefix}_planned.csv")
    else:
        print("Not enough complete rows (need ≥ 3).")

    # -------- Actual pipeline (independent) --------
    dA = build_mode_dataset(milestones, "actual")
    jA = dA.merge(approvals, on="study", how="inner").dropna(subset=["APPROVAL_MIN"]).reset_index(drop=True)

    print("\n==== ACTUAL ====")
    if len(jA) >= 3:
        fitA = fit_constant_gap(jA)
        print(f"Studies used (n): {fitA['n']}")
        print(f"GAP_hat (days): {fitA['gap_hat_days']:.2f} | sd: {fitA['gap_sd_days']:.2f} | "
              f"t_0.70: {fitA['t_0_70']:.4f} | pred_sd: {fitA['pred_sd_days']:.2f}")
        print(f"Anchor date: {fitA['anchor'].date()}")
        print("Prediction equation:")
        print("  approval_hat = anchor + timedelta( A_days + GAP_hat )")
        print("  A_days = (LSLV − LSSE) + (FSSE − anchor)  [days]")
        print("  LB70 = anchor + timedelta( A_days + GAP_hat − t_{0.70,df} * sd * sqrt(1 + 1/n) )")

        preds = []
        for r in jA.itertuples(index=False):
            pt, lb = fitA["predict_from_A_days"](r.A_days)
            preds.append(dict(study=r.study, A_days=r.A_days,
                              pred_approval=str(pt.date()),
                              LB70_approval=str(lb.date()),
                              actual_approval=str(r.APPROVAL_MIN.date())))
        pd.DataFrame(preds).to_csv(f"{out_prefix}_actual.csv", index=False)
        print(f"Per-study predictions saved: {out_prefix}_actual.csv")
    else:
        print("Not enough complete rows (need ≥ 3).")

    # -------- console check to prove independence --------
    nP = incl["planned_included"].sum()
    nA = incl["actual_included"].sum()
    both = ((incl["planned_included"]) & (incl["actual_included"])).sum()
    p_only = (incl["planned_included"] & ~incl["actual_included"]).sum()
    a_only = (~incl["planned_included"] & incl["actual_included"]).sum()
    neither = (~incl["planned_included"] & ~incl["actual_included"]).sum()
    print("\n==== INCLUSION MATRIX ====")
    print(f"Planned included: {nP} | Actual included: {nA}")
    print(f"Both: {both} | Planned-only: {p_only} | Actual-only: {a_only} | Neither: {neither}")
    print("Full per-study inclusion written to: inclusion_matrix.csv")
    print("Detailed audits written to: audit_planned.csv, audit_actual.csv")

# -------------------- EXECUTE --------------------
if __name__ == "__main__":
    run(MILESTONES_CSV, APPROVALS_CSV)
