"""
Week 3 - Exploratory Data Analysis and Visualization
Logistics Data Analytics Internship

Input:
    data/Food_Delivery_Times_Cleaned_Week2.csv

Output:
    charts/week3/
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "Food_Delivery_Times_Cleaned_Week2.csv"
CHART_DIR = ROOT / "charts" / "week3"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_FILE)

    print("Shape:", df.shape)
    print("Average delivery time:", round(df["Delivery_Time_min"].mean(), 2))
    print("Median delivery time:", round(df["Delivery_Time_min"].median(), 2))
    print("90th percentile:", round(df["Delivery_Time_min"].quantile(0.90), 2))
    print(
        "Distance correlation:",
        round(df["Distance_km"].corr(df["Delivery_Time_min"]), 3),
    )
    print(
        "Preparation correlation:",
        round(df["Preparation_Time_min"].corr(df["Delivery_Time_min"]), 3),
    )

    # 1. Delivery-time distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Delivery_Time_min"], kde=True, ax=ax)
    ax.set_title("Delivery Time Distribution")
    ax.set_xlabel("Delivery Time (minutes)")
    save(fig, "01_delivery_time_distribution.png")

    # 2. Distance vs delivery time
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="Distance_km",
        y="Delivery_Time_min",
        ax=ax,
        alpha=0.65,
    )
    ax.set_title("Distance vs Delivery Time")
    save(fig, "02_distance_vs_delivery_time.png")

    # 3. Traffic
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="Traffic_Level",
        y="Delivery_Time_min",
        estimator="mean",
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Average Delivery Time by Traffic Level")
    save(fig, "03_traffic_average_delivery.png")

    # 4. Weather
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="Weather",
        y="Delivery_Time_min",
        estimator="mean",
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Average Delivery Time by Weather")
    save(fig, "04_weather_average_delivery.png")

    # 5. Time of day
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="Time_of_Day",
        y="Delivery_Time_min",
        estimator="mean",
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Average Delivery Time by Time of Day")
    save(fig, "05_time_of_day_average_delivery.png")

    # 6. Vehicle type
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="Vehicle_Type",
        y="Delivery_Time_min",
        estimator="mean",
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Average Delivery Time by Vehicle Type")
    save(fig, "06_vehicle_average_delivery.png")

    # 7. Correlation matrix
    numeric = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    save(fig, "07_correlation_matrix.png")

    # 8. Preparation time vs delivery time
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="Preparation_Time_min",
        y="Delivery_Time_min",
        ax=ax,
        alpha=0.65,
    )
    ax.set_title("Preparation Time vs Delivery Time")
    save(fig, "08_preparation_vs_delivery_time.png")

    print(f"Charts saved to: {CHART_DIR}")


if __name__ == "__main__":
    main()
