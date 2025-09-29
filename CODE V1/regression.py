import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def analyze_and_visualize():


    try:
        df = pd.read_csv('Analytical Method for FSLV Prediction - FSLV.csv')
    except FileNotFoundError:
        print(" Error: 'Analytical Method for FSLV Prediction - FSLV.csv' not found.")
        return

    df.rename(columns={
        'Study #': 'study_number',
        'First Estimated PTA Treatment Date': 'first_estimated_pta_treatment_date'
    }, inplace=True)

    all_cols = df.columns
    date_columns_to_check = [col for col in all_cols if 'Unnamed' not in col and col not in ['study_number', 'FSLV',
                                                                                             'first_estimated_pta_treatment_date']]

    for col in ['FSLV', 'first_estimated_pta_treatment_date'] + date_columns_to_check:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')


    df['closest_to_FSLV_col'], df['closest_to_PTA_col'] = pd.NA, pd.NA
    if df['FSLV'].notna().any():
        diffs_fslv = df[date_columns_to_check].sub(df['FSLV'], axis=0).abs()
        valid_rows = diffs_fslv.dropna(how='all').index
        if not valid_rows.empty:
            df.loc[valid_rows, 'closest_to_FSLV_col'] = diffs_fslv.loc[valid_rows].idxmin(axis=1, skipna=True)

    if df['first_estimated_pta_treatment_date'].notna().any():
        diffs_pta = df[date_columns_to_check].sub(df['first_estimated_pta_treatment_date'], axis=0).abs()
        valid_rows_pta = diffs_pta.dropna(how='all').index
        if not valid_rows_pta.empty:
            df.loc[valid_rows_pta, 'closest_to_PTA_col'] = diffs_pta.loc[valid_rows_pta].idxmin(axis=1, skipna=True)


    print("\n" + "=" * 50)
    print(" Regression Analysis and R-Squared Statistics")
    print("=" * 50)
    sns.set_style("whitegrid")


    fslv_reg_data = df.dropna(subset=['FSLV', 'closest_to_FSLV_col']).copy()
    if not fslv_reg_data.empty:
        predictor_dates = [fslv_reg_data.loc[idx, col] for idx, col in fslv_reg_data['closest_to_FSLV_col'].items()]
        fslv_reg_data['best_fslv_predictor_date'] = predictor_dates

        X_fslv = fslv_reg_data['best_fslv_predictor_date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        y_fslv = fslv_reg_data['FSLV'].map(pd.Timestamp.toordinal).values

        model_fslv = LinearRegression().fit(X_fslv, y_fslv)
        r2_fslv = r2_score(y_fslv, model_fslv.predict(X_fslv))
        slope_fslv = model_fslv.coef_[0]
        intercept_fslv = model_fslv.intercept_

        print(f"\n--- FSLV Regression Results ---")
        print(f"R-squared (R²): {r2_fslv:.4f}")
        print(f"Model Equation: Predicted_FSLV = {slope_fslv:.4f} * (Predictor_Date) + {intercept_fslv:.2f}")

        fslv_reg_data['X_ordinal'] = X_fslv
        fslv_reg_data['y_ordinal'] = y_fslv
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.regplot(x='X_ordinal', y='y_ordinal', data=fslv_reg_data, ax=ax, line_kws={"color": "red"})

        def format_func(x, pos):
            return pd.to_datetime(pd.Timestamp.fromordinal(int(x))).strftime('%Y-%m-%d')

        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')


        eq_text = f'Predicted FSLV = {slope_fslv:.4f} * (Predictor) + {intercept_fslv:.2f}\n$R^2$ = {r2_fslv:.4f}'
        ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.6))

        ax.set_title("FSLV vs. Best Predictor Date", fontsize=16)
        ax.set_xlabel("Best Predictor Date")
        ax.set_ylabel("Actual FSLV Date")
        plt.tight_layout()
        plt.savefig('fslv_regression_plot.png')
        print("Saved 'fslv_regression_plot.png'")
        plt.close()


    pta_reg_data = df.dropna(subset=['first_estimated_pta_treatment_date', 'closest_to_PTA_col']).copy()
    if not pta_reg_data.empty:
        predictor_dates_pta = [pta_reg_data.loc[idx, col] for idx, col in pta_reg_data['closest_to_PTA_col'].items()]
        pta_reg_data['best_pta_predictor_date'] = predictor_dates_pta

        X_pta = pta_reg_data['best_pta_predictor_date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        y_pta = pta_reg_data['first_estimated_pta_treatment_date'].map(pd.Timestamp.toordinal).values

        model_pta = LinearRegression().fit(X_pta, y_pta)
        r2_pta = r2_score(y_pta, model_pta.predict(X_pta))
        slope_pta = model_pta.coef_[0]
        intercept_pta = model_pta.intercept_

        print(f"\n--- PTA Regression Results ---")
        print(f"R-squared (R²): {r2_pta:.4f}")
        print(f"Model Equation: Predicted_PTA = {slope_pta:.4f} * (Predictor_Date) + {intercept_pta:.2f}")

        pta_reg_data['X_ordinal'] = X_pta
        pta_reg_data['y_ordinal'] = y_pta
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.regplot(x='X_ordinal', y='y_ordinal', data=pta_reg_data, ax=ax, line_kws={"color": "blue"})

        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')


        eq_text_pta = f'Predicted PTA = {slope_pta:.4f} * (Predictor) + {intercept_pta:.2f}\n$R^2$ = {r2_pta:.4f}'
        ax.text(0.05, 0.95, eq_text_pta, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.6))

        ax.set_title("PTA Date vs. Best Predictor Date", fontsize=16)
        ax.set_xlabel("Best Predictor Date")
        ax.set_ylabel("Actual PTA Date")
        plt.tight_layout()
        plt.savefig('pta_regression_plot.png')
        print(" Saved 'pta_regression_plot.png'")
        plt.close()

    df.to_csv('analysis_results_with_regression.csv', index=False)
    print("\n Full results saved to 'analysis_results_with_regression.csv'")


if __name__ == '__main__':
    analyze_and_visualize()