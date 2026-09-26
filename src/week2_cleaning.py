"""
Week 2 - Data Cleaning and Preprocessing
Logistics Data Analytics Internship

Input:
    Food_Delivery_Times.csv

Output:
    Food_Delivery_Times_Cleaned_Week2.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INPUT_FILE = DATA_DIR / "Food_Delivery_Times.csv"
OUTPUT_FILE = DATA_DIR / "Food_Delivery_Times_Cleaned_Week2.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    print(f"Original shape: {df.shape}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Missing cells: {int(df.isna().sum().sum())}")

    # Remove exact duplicate rows.
    df = df.drop_duplicates().copy()

    categorical_cols = ["Weather", "Traffic_Level", "Time_of_Day"]
    numeric_impute_cols = ["Courier_Experience_yrs"]

    # Preserve information about which values were originally missing.
    for col in categorical_cols + numeric_impute_cols:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)

    # Mode imputation for categorical variables.
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

    # Median imputation for courier experience.
    df["Courier_Experience_yrs"] = df["Courier_Experience_yrs"].fillna(
        df["Courier_Experience_yrs"].median()
    )

    # Flag IQR outliers in the delivery-time target.
    q1 = df["Delivery_Time_min"].quantile(0.25)
    q3 = df["Delivery_Time_min"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df["Delivery_Time_Outlier_Flag"] = (
        (df["Delivery_Time_min"] < lower) |
        (df["Delivery_Time_min"] > upper)
    ).astype(int)

    # Standardize selected numerical predictors.
    scale_cols = [
        "Distance_km",
        "Preparation_Time_min",
        "Courier_Experience_yrs",
    ]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Cleaned shape: {df.shape}")
    print(f"Delivery-time IQR outliers flagged: {df['Delivery_Time_Outlier_Flag'].sum()}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
