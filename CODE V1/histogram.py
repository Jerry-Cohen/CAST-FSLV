import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_date_differences():

    try:
        df = pd.read_csv('Analytical Method for FSLV Prediction - FSLV.csv')
    except FileNotFoundError:
        print("Error: 'Analytical Method for FSLV Prediction - FSLV.csv' not found.")
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


    df['closest_to_FSLV_col'] = pd.NA
    if df['FSLV'].notna().any():
        diffs_fslv = df[date_columns_to_check].sub(df['FSLV'], axis=0).abs()
        valid_rows_fslv = diffs_fslv.dropna(how='all').index
        if not valid_rows_fslv.empty:
            df.loc[valid_rows_fslv, 'closest_to_FSLV_col'] = diffs_fslv.loc[valid_rows_fslv].idxmin(axis=1, skipna=True)

    df['closest_to_PTA_col'] = pd.NA
    if df['first_estimated_pta_treatment_date'].notna().any():
        diffs_pta = df[date_columns_to_check].sub(df['first_estimated_pta_treatment_date'], axis=0).abs()
        valid_rows_pta = diffs_pta.dropna(how='all').index
        if not valid_rows_pta.empty:
            df.loc[valid_rows_pta, 'closest_to_PTA_col'] = diffs_pta.loc[valid_rows_pta].idxmin(axis=1, skipna=True)


    print("\n" + "=" * 50)
    print(" Analysis of FSLV Time Difference (Duration)")
    print("=" * 50)
    fslv_diff_data = df.dropna(subset=['FSLV', 'closest_to_FSLV_col']).copy()
    if not fslv_diff_data.empty:
        predictor_dates_fslv = [fslv_diff_data.loc[idx, col] for idx, col in
                                fslv_diff_data['closest_to_FSLV_col'].items()]
        fslv_diff_data['difference_in_days'] = (fslv_diff_data['FSLV'] - pd.to_datetime(predictor_dates_fslv)).dt.days

        mean_diff_fslv = fslv_diff_data['difference_in_days'].mean()
        std_diff_fslv = fslv_diff_data['difference_in_days'].std()
        median_diff_fslv = fslv_diff_data['difference_in_days'].median()

        print(f"Average Duration: {mean_diff_fslv:.2f} days")
        print(f"Standard Deviation: {std_diff_fslv:.2f} days")
        print(f"Median Duration: {median_diff_fslv:.0f} days")

        plt.figure(figsize=(12, 7))
        sns.histplot(data=fslv_diff_data, x='difference_in_days', kde=True, bins=30)
        plt.axvline(mean_diff_fslv, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_diff_fslv:.2f}')
        plt.axvline(median_diff_fslv, color='green', linestyle='-', linewidth=2,
                    label=f'Median: {median_diff_fslv:.0f}')
        plt.title("Distribution of FSLV Time Difference (Duration)", fontsize=16)
        plt.xlabel("Difference in Days (Actual FSLV - Predictor Date)")
        plt.legend()
        plt.tight_layout()
        plt.savefig('fslv_difference_histogram.png')
        print(" Saved 'fslv_difference_histogram.png'")
        plt.close()


    print("\n" + "=" * 50)
    print("Analysis of PTA Time Difference (Duration)")
    print("=" * 50)
    pta_diff_data = df.dropna(subset=['first_estimated_pta_treatment_date', 'closest_to_PTA_col']).copy()
    if not pta_diff_data.empty:
        predictor_dates_pta = [pta_diff_data.loc[idx, col] for idx, col in pta_diff_data['closest_to_PTA_col'].items()]
        pta_diff_data['difference_in_days'] = (
                    pta_diff_data['first_estimated_pta_treatment_date'] - pd.to_datetime(predictor_dates_pta)).dt.days

        mean_diff_pta = pta_diff_data['difference_in_days'].mean()
        std_diff_pta = pta_diff_data['difference_in_days'].std()
        median_diff_pta = pta_diff_data['difference_in_days'].median()

        print(f"Average Duration: {mean_diff_pta:.2f} days")
        print(f"Standard Deviation: {std_diff_pta:.2f} days")
        print(f"Median Duration: {median_diff_pta:.0f} days")

        plt.figure(figsize=(12, 7))
        sns.histplot(data=pta_diff_data, x='difference_in_days', kde=True, bins=30, color='purple')
        plt.axvline(mean_diff_pta, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_diff_pta:.2f}')
        plt.axvline(median_diff_pta, color='green', linestyle='-', linewidth=2, label=f'Median: {median_diff_pta:.0f}')
        plt.title("Distribution of PTA Time Difference (Duration)", fontsize=16)
        plt.xlabel("Difference in Days (Actual PTA - Predictor Date)")
        plt.legend()
        plt.tight_layout()
        plt.savefig('pta_difference_histogram.png')
        print(" Saved 'pta_difference_histogram.png'")
        plt.close()


if __name__ == '__main__':
    analyze_date_differences()