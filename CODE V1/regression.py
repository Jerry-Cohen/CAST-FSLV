import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ------------------------------------------------------------
# Utility: nice date tick formatter for ordinal axes
# ------------------------------------------------------------
def _fmt_ordinal(v, pos):
    try:
        return pd.Timestamp.fromordinal(int(v)).strftime('%Y-%m-%d')
    except Exception:
        return ""

def analyze_and_visualize():
    # -------------------- Load --------------------
    try:
        df = pd.read_csv('Analytical Method for FSLV Prediction - FSLV.csv')
    except FileNotFoundError:
        print("Error: 'Analytical Method for FSLV Prediction - FSLV.csv' not found.")
        return

    # Normalize column names we depend on
    df.rename(columns={
        'Study #': 'study_number',
        'First Estimated PTA Treatment Date': 'first_estimated_pta_treatment_date'
    }, inplace=True)

    # Identify candidate date columns (exclude ID + targets)
    all_cols = df.columns.tolist()
    date_candidates = [c for c in all_cols
                       if ('Unnamed' not in c)
                       and (c not in ['study_number', 'FSLV', 'first_estimated_pta_treatment_date'])]

    # Parse dates robustly
    for col in ['FSLV', 'first_estimated_pta_treatment_date'] + date_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    print("\n" + "=" * 50)
    print(" Regression Analysis and R-Squared Statistics")
    print("=" * 50)
    sns.set_style("whitegrid")

    # ============================================================
    # FSLV REGRESSION (keeps your original idea; no CI needed here)
    # ============================================================
    # rows with FSLV and at least one candidate date present
    mask_valid_fslv = df['FSLV'].notna()
    mask_any_cand = df[date_candidates].notna().any(axis=1) if date_candidates else False
    fslv_reg_data = df[mask_valid_fslv & mask_any_cand].copy()

    print(f"\nFSLV rows eligible for regression: {len(fslv_reg_data)}")
    if len(fslv_reg_data) >= 3 and date_candidates:
        # pick per-row predictor closest to FSLV
        diffs_fslv = (fslv_reg_data[date_candidates]
                      .sub(fslv_reg_data['FSLV'], axis=0).abs())
        best_cols_fslv = diffs_fslv.idxmin(axis=1, skipna=True)
        best_pred_fslv = best_cols_fslv.to_frame('col').assign(row=diffs_fslv.index).apply(
            lambda r: fslv_reg_data.at[r['row'], r['col']], axis=1
        )

        valid_idx_fslv = best_pred_fslv.index[best_pred_fslv.notna()]
        fslv_reg_data = fslv_reg_data.loc[valid_idx_fslv].copy()
        best_pred_fslv = best_pred_fslv.loc[valid_idx_fslv]

        X_fslv = best_pred_fslv.map(pd.Timestamp.toordinal).to_numpy(dtype=float).reshape(-1, 1)
        y_fslv = fslv_reg_data['FSLV'].map(pd.Timestamp.toordinal).to_numpy(dtype=float)

        if X_fslv.shape[0] >= 3 and np.var(X_fslv) > 0:
            model_fslv = LinearRegression().fit(X_fslv, y_fslv)
            yhat_fslv = model_fslv.predict(X_fslv).flatten()
            r2_fslv = r2_score(y_fslv, yhat_fslv)
            slope_fslv = float(model_fslv.coef_[0])
            intercept_fslv = float(model_fslv.intercept_)

            print(f"\n--- FSLV Regression Results ---")
            print(f"R-squared (R²): {r2_fslv:.4f}")
            print(f"Model Equation: Predicted_FSLV = {slope_fslv:.4f} * (Predictor_Date) + {intercept_fslv:.2f}")

            # Plot (FSLV)
            plot_df_fslv = pd.DataFrame({'X_ordinal': X_fslv.flatten(), 'y_ordinal': y_fslv})
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.regplot(x='X_ordinal', y='y_ordinal', data=plot_df_fslv, ax=ax, line_kws={"color": "red"})
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_ordinal))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_ordinal))
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            eq_text = f'Predicted FSLV = {slope_fslv:.4f} * (Predictor) + {intercept_fslv:.2f}\n$R^2$ = {r2_fslv:.4f}'
            ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, fontsize=12,
                    va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.6))
            ax.set_title("FSLV vs. Best Predictor Date", fontsize=16)
            ax.set_xlabel("Best Predictor Date")
            ax.set_ylabel("Actual FSLV Date")
            plt.tight_layout()
            plt.savefig('fslv_regression_plot.png')
            print("Saved 'fslv_regression_plot.png'")
            plt.close()
        else:
            print("Not enough variation or rows to fit FSLV regression.")
    else:
        print("No eligible FSLV rows (missing FSLV dates or predictors).")

    # ============================================================
    # PTA REGRESSION + one-sided 70% LOWER CI (earliest)
    # ============================================================
    mask_valid_pta = df['first_estimated_pta_treatment_date'].notna()
    mask_any_cand = df[date_candidates].notna().any(axis=1) if date_candidates else False
    pta_reg_data = df[mask_valid_pta & mask_any_cand].copy()

    print(f"\nPTA rows eligible for regression: {len(pta_reg_data)}")
    if len(pta_reg_data) >= 3 and date_candidates:
        # per-row predictor: candidate closest to PTA
        diffs_pta = (pta_reg_data[date_candidates]
                     .sub(pta_reg_data['first_estimated_pta_treatment_date'], axis=0).abs())
        best_cols_pta = diffs_pta.idxmin(axis=1, skipna=True)
        best_pred_pta = best_cols_pta.to_frame('col').assign(row=diffs_pta.index).apply(
            lambda r: pta_reg_data.at[r['row'], r['col']], axis=1
        )

        valid_idx_pta = best_pred_pta.index[best_pred_pta.notna()]
        pta_reg_data = pta_reg_data.loc[valid_idx_pta].copy()
        best_pred_pta = best_pred_pta.loc[valid_idx_pta]

        X_pta = best_pred_pta.map(pd.Timestamp.toordinal).to_numpy(dtype=float).reshape(-1, 1)
        y_pta = pta_reg_data['first_estimated_pta_treatment_date'].map(pd.Timestamp.toordinal).to_numpy(dtype=float)

        if X_pta.shape[0] >= 3 and np.var(X_pta) > 0:
            model_pta = LinearRegression().fit(X_pta, y_pta)
            yhat_pta = model_pta.predict(X_pta).flatten()
            r2_pta = r2_score(y_pta, yhat_pta)
            slope_pta = float(model_pta.coef_[0])
            intercept_pta = float(model_pta.intercept_)

            print(f"\n--- PTA Regression Results ---")
            print(f"R-squared (R²): {r2_pta:.4f}")
            print(f"Model Equation: Predicted_PTA = {slope_pta:.4f} * (Predictor_Date) + {intercept_pta:.2f}")

            # ----- One-sided 70% LOWER CI for the mean (earliest) -----
            x = X_pta.flatten()
            y = y_pta
            n = x.size
            xbar = float(x.mean())
            resid = y - yhat_pta
            s = float(np.sqrt(np.sum(resid**2) / (n - 2)))
            Sxx = float(np.sum((x - xbar)**2))
            if n <= 2 or np.isclose(Sxx, 0):
                print("Not enough variation/points to compute PTA CI. Skipping lower bound.")
                lower70_ord = None
                lower70_dates = None
            else:
                try:
                    from scipy.stats import t
                    tcrit = float(t.ppf(0.70, df=n-2))  # one-sided 70%
                except Exception:
                    tcrit = 0.5244005127080409          # normal approx

                def lower_mean_70(x0: float) -> float:
                    y0 = float(model_pta.predict(np.array([[x0]])).item())
                    band = tcrit * s * np.sqrt(1.0/n + (x0 - xbar)**2 / Sxx)
                    return y0 - band

                lower70_ord = np.array([lower_mean_70(xx) for xx in x], dtype=float)

                # Build a Series (has .iloc) and aligns to the same index as pta_reg_data
                lower70_dates = pd.Series(
                    [pd.Timestamp.fromordinal(int(round(v))) for v in lower70_ord],
                    index=pta_reg_data.index,
                    dtype='datetime64[ns]'
                )

                # Report cohort earliest
                i_min = int(np.argmin(lower70_ord))
                earliest_date = lower70_dates.iloc[i_min]
                print(f"Earliest one-sided 70% LOWER (mean) PTA across cohort: {earliest_date:%Y-%m-%d}")

                # attach to df for the matched rows
                df.loc[pta_reg_data.index, 'pta_lower70_date'] = lower70_dates.values

                # also save a compact file
                out_cols = ['study_number', 'first_estimated_pta_treatment_date']
                if 'study_number' not in pta_reg_data.columns:
                    # keep index if no study_number
                    pta_reg_data = pta_reg_data.copy()
                    pta_reg_data['row_index'] = pta_reg_data.index.astype(int)
                    out_cols = ['row_index', 'first_estimated_pta_treatment_date']
                tmp_out = pta_reg_data.copy()
                tmp_out['best_pta_predictor_date'] = pd.to_datetime(best_pred_pta.values)
                tmp_out['pta_lower70_date'] = lower70_dates.values
                tmp_out[out_cols + ['best_pta_predictor_date', 'pta_lower70_date']].to_csv('pta_with_lower70.csv', index=False)
                print("Saved 'pta_with_lower70.csv' with per-study earliest (one-sided 70%) PTA dates.")

            # ----- Plot (PTA)
            plot_df_pta = pd.DataFrame({'X_ordinal': x, 'y_ordinal': y})
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.regplot(x='X_ordinal', y='y_ordinal', data=plot_df_pta, ax=ax, line_kws={"color": "blue"})
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_ordinal))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_ordinal))
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            eq_text_pta = f'Predicted PTA = {slope_pta:.4f} * (Predictor) + {intercept_pta:.2f}\n$R^2$ = {r2_pta:.4f}'
            ax.text(0.05, 0.95, eq_text_pta, transform=ax.transAxes, fontsize=12,
                    va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.6))
            ax.set_title("PTA Date vs. Best Predictor Date", fontsize=16)
            ax.set_xlabel("Best Predictor Date")
            ax.set_ylabel("Estimated PTA Date")
            plt.tight_layout()
            plt.savefig('pta_regression_plot.png')
            print("Saved 'pta_regression_plot.png'")
            plt.close()
        else:
            print("Not enough variation or rows to fit PTA regression.")
    else:
        print("No eligible PTA rows (missing PTA dates or predictors).")

    # Save the master dataframe with any added outputs
    df.to_csv('analysis_results_with_regression.csv', index=False)
    print("\nFull results saved to 'analysis_results_with_regression.csv'")

if __name__ == '__main__':
    analyze_and_visualize()
