import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_visualize():

    df = pd.read_csv('Analytical Method for FSLV Prediction - FSLV.csv')


    df.rename(columns={
        'Study #': 'study_number',
        'First Estimated PTA Treatment Date': 'first_estimated_pta_treatment_date'
    }, inplace=True)

    all_cols = df.columns
    date_columns_to_check = [
        col for col in all_cols
        if 'Unnamed' not in col and col not in [
            'study_number', 'FSLV', 'first_estimated_pta_treatment_date'
        ]
    ]

    reference_cols = ['FSLV', 'first_estimated_pta_treatment_date']
    for col in reference_cols + date_columns_to_check:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')


    df['closest_to_FSLV'], df['closest_to_PTA'] = pd.NaT, pd.NaT

    # Analysis for 'FSLV'
    fslv_not_na = df['FSLV'].notna()
    if fslv_not_na.any():
        diffs_fslv = df.loc[fslv_not_na, date_columns_to_check].sub(df.loc[fslv_not_na, 'FSLV'], axis=0).abs()
        valid_rows = diffs_fslv.dropna(how='all').index
        if not valid_rows.empty:
            closest_fslv = diffs_fslv.loc[valid_rows].idxmin(axis=1, skipna=True)
            df.loc[closest_fslv.index, 'closest_to_FSLV'] = closest_fslv

        fslv_diff_days = diffs_fslv.apply(lambda x: x.dt.days)


    pta_not_na = df['first_estimated_pta_treatment_date'].notna()
    if pta_not_na.any():
        diffs_pta = df.loc[pta_not_na, date_columns_to_check].sub(df.loc[pta_not_na, 'first_estimated_pta_treatment_date'], axis=0).abs()
        valid_rows_pta = diffs_pta.dropna(how='all').index
        if not valid_rows_pta.empty:
            closest_pta = diffs_pta.loc[valid_rows_pta].idxmin(axis=1, skipna=True)
            df.loc[closest_pta.index, 'closest_to_PTA'] = closest_pta

        pta_diff_days = diffs_pta.apply(lambda x: x.dt.days)


    print("\n" + "=" * 60)
    print("📊 Descriptive Statistics for Time Differences (in Days)")
    print("=" * 60)
    if 'fslv_diff_days' in locals():
        print("\n--- Statistics for Differences from 'FSLV' ---")
        print(fslv_diff_days.describe().transpose().to_string())
    if 'pta_diff_days' in locals():
        print("\n--- Statistics for Differences from 'first estimated PTA treatment date' ---")
        print(pta_diff_days.describe().transpose().to_string())


    print("\n" + "=" * 50)
    print("🎨 Generating and saving visualizations...")
    print("=" * 50)
    sns.set_style("whitegrid")


    if 'fslv_diff_days' in locals():

        plt.figure(figsize=(16, 8))
        mean_diffs_fslv = fslv_diff_days.mean().sort_values()
        sns.barplot(x=mean_diffs_fslv.index, y=mean_diffs_fslv.values, palette="crest")
        plt.title("Average Time Difference from 'FSLV'", fontsize=16, weight='bold')
        plt.ylabel("Mean Difference in Days", fontsize=12)
        plt.tight_layout()
        plt.savefig('fslv_mean_difference_bar.png')
        print("✅ Saved 'fslv_mean_difference_bar.png'")
        plt.close()


        plt.figure(figsize=(16, 8))
        sns.boxplot(data=fslv_diff_days, palette="coolwarm")
        y_limit = np.nanpercentile(fslv_diff_days.values, 99)
        plt.ylim(0, y_limit * 1.05)
        plt.title("Distribution of Time Differences from 'FSLV' (Zoomed In)", fontsize=16, weight='bold')
        plt.ylabel("Absolute Difference in Days", fontsize=12)
        plt.tight_layout()
        plt.savefig('fslv_difference_boxplot_zoomed.png')
        print("✅ Saved 'fslv_difference_boxplot_zoomed.png'")
        plt.close()


    if 'pta_diff_days' in locals():

        plt.figure(figsize=(16, 8))
        mean_diffs_pta = pta_diff_days.mean().sort_values()
        sns.barplot(x=mean_diffs_pta.index, y=mean_diffs_pta.values, palette="rocket_r")
        plt.title("Average Time Difference from 'First Estimated PTA Date'", fontsize=16, weight='bold')
        plt.ylabel("Mean Difference in Days", fontsize=12)
        plt.tight_layout()
        plt.savefig('pta_mean_difference_bar.png')
        print("✅ Saved 'pta_mean_difference_bar.png'")
        plt.close()


        plt.figure(figsize=(16, 8))
        sns.boxplot(data=pta_diff_days, palette='magma')
        y_limit_pta = np.nanpercentile(pta_diff_days.values, 99)
        plt.ylim(0, y_limit_pta * 1.05)
        plt.title("Distribution of Time Differences from 'First Estimated PTA Date' (Zoomed In)", fontsize=16, weight='bold')
        plt.ylabel("Absolute Difference in Days", fontsize=12)
        plt.tight_layout()
        plt.savefig('pta_difference_boxplot_zoomed.png')
        print("✅ Saved 'pta_difference_boxplot_zoomed.png'")
        plt.close()


    df.to_csv('analysis_results_with_stats.csv', index=False)
    print("\n💾 Full results saved to 'analysis_results_with_stats.csv'")


if __name__ == '__main__':
    analyze_and_visualize()