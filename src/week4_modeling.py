"""
Week 4 - Predictive Modeling and Optimization
Logistics Data Analytics Internship

Input:
    data/Food_Delivery_Times_Cleaned_Week2.csv

Outputs:
    results/Week_4_Model_Comparison.csv
    results/Week_4_Cross_Validation_Results.csv
    charts/week4/01_actual_vs_predicted.png
    charts/week4/02_feature_importance.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "Food_Delivery_Times_Cleaned_Week2.csv"
RESULT_DIR = ROOT / "results"
CHART_DIR = ROOT / "charts" / "week4"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def main():
    df = pd.read_csv(DATA_FILE)

    target = "Delivery_Time_min"
    features = [
        "Distance_km",
        "Weather",
        "Traffic_Level",
        "Time_of_Day",
        "Vehicle_Type",
        "Preparation_Time_min",
        "Courier_Experience_yrs",
    ]

    X = df[features]
    y = df[target]

    numeric_features = [
        "Distance_km",
        "Preparation_Time_min",
        "Courier_Experience_yrs",
    ]
    categorical_features = [
        "Weather",
        "Traffic_Level",
        "Time_of_Day",
        "Vehicle_Type",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }

    rows = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numeric_features, categorical_features),
                ),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": mean_squared_error(y_test, pred, squared=False),
                "R2": r2_score(y_test, pred),
            }
        )
        fitted_pipelines[name] = pipe

    comparison = pd.DataFrame(rows).sort_values("RMSE")
    comparison.to_csv(RESULT_DIR / "Week_4_Model_Comparison.csv", index=False)

    # Five-fold cross-validation.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numeric_features, categorical_features),
                ),
                ("model", model),
            ]
        )

        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)

        cv_rows.append(
            {
                "Model": name,
                "CV_MAE": -scores["test_MAE"].mean(),
                "CV_RMSE": -scores["test_RMSE"].mean(),
                "CV_R2": scores["test_R2"].mean(),
            }
        )

    cv_results = pd.DataFrame(cv_rows).sort_values("CV_RMSE")
    cv_results.to_csv(
        RESULT_DIR / "Week_4_Cross_Validation_Results.csv",
        index=False,
    )

    # Select the best held-out test model by RMSE.
    best_name = comparison.iloc[0]["Model"]
    best_pipe = fitted_pipelines[best_name]
    best_pred = best_pipe.predict(X_test)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, best_pred, alpha=0.7)
    low = min(y_test.min(), best_pred.min())
    high = max(y_test.max(), best_pred.max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual Delivery Time (minutes)")
    ax.set_ylabel("Predicted Delivery Time (minutes)")
    ax.set_title(f"Actual vs Predicted - {best_name}")
    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "01_actual_vs_predicted.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Coefficient magnitude for Linear Regression, useful for interpretation.
    linear_pipe = fitted_pipelines["Linear Regression"]
    preprocessor = linear_pipe.named_steps["preprocessor"]
    model = linear_pipe.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = np.abs(model.coef_)

    importance = (
        pd.DataFrame({"Feature": feature_names, "Importance": coefficients})
        .sort_values("Importance", ascending=False)
        .head(15)
        .sort_values("Importance")
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance["Feature"], importance["Importance"])
    ax.set_xlabel("Absolute coefficient magnitude")
    ax.set_title("Top Linear Regression Feature Contributions")
    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "02_feature_importance.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nHeld-out test results:")
    print(comparison.to_string(index=False))
    print("\nFive-fold cross-validation:")
    print(cv_results.to_string(index=False))
    print(f"\nBest test RMSE model: {best_name}")


if __name__ == "__main__":
    main()
