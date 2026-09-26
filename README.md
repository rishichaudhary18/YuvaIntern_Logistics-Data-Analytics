# Logistics Data Analytics Internship Project

## 📌 Project Overview

This repository contains my 4-week Logistics Data Analytics internship project. The project focuses on analyzing food delivery operations, identifying important factors affecting delivery time, performing exploratory data analysis, developing predictive models, and providing data-driven recommendations for logistics optimization.

The project follows a complete analytics workflow:

**Data Collection → Data Cleaning → Exploratory Data Analysis → Visualization → Predictive Modeling → Optimization Recommendations**

## 📅 Weekly Work

### Week 1 — Strategic Planning & Data Exploration

* Defined the logistics delivery-time prediction problem.
* Identified important logistics KPIs.
* Researched relevant logistics analytics approaches.
* Explored regression, clustering, and optimization concepts.
* Developed a roadmap for the complete analytics project.

### Week 2 — Data Collection, Cleaning & Preprocessing

* Used a public food delivery dataset containing 1,000 records.
* Identified missing values, duplicate records, and numerical outliers.
* Handled missing categorical and numerical values.
* Added missingness and outlier indicators.
* Standardized selected numerical features.
* Created a cleaned dataset for subsequent analysis.

### Week 3 — Advanced Data Analysis & Visualization

Performed exploratory data analysis to understand factors affecting delivery time.

Key findings included:

* Average delivery time: **56.73 minutes**
* Median delivery time: **55.50 minutes**
* 90th percentile delivery time: **86 minutes**
* Distance and delivery time showed a Pearson correlation of approximately **0.781**.
* Preparation time and delivery time showed a correlation of approximately **0.307**.

Multiple visualizations were created to analyze delivery-time distributions, traffic, weather, time of day, vehicle type, distance, preparation time, and feature correlations.

### Week 4 — Predictive Modeling & Optimization

The target variable was:

`Delivery_Time_min`

Features included:

* Distance
* Weather
* Traffic level
* Time of day
* Vehicle type
* Preparation time
* Courier experience

Four regression models were evaluated:

| Model             |   MAE |   RMSE |    R² |
| ----------------- | ----: | -----: | ----: |
| Linear Regression | 5.899 |  8.826 | 0.826 |
| Ridge Regression  | 5.905 |  8.829 | 0.826 |
| Gradient Boosting | 6.370 |  9.196 | 0.811 |
| Random Forest     | 7.498 | 10.054 | 0.774 |

Linear Regression achieved the best performance on the held-out test set, with an RMSE of approximately **8.83 minutes** and an R² of approximately **0.826**.

Five-fold cross-validation was also performed to evaluate model consistency.

## 📊 Dataset

The project uses a food delivery dataset containing information about:

* Order ID
* Distance
* Weather
* Traffic level
* Time of day
* Vehicle type
* Preparation time
* Courier experience
* Delivery time

The original dataset contains **1,000 records and 9 columns**.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook / Python
* Microsoft Word
* Git & GitHub

## 📈 Optimization Recommendations

Based on the analysis, the project proposes:

* Dynamic delivery-time estimation
* Traffic-aware delivery planning
* Workload-aware courier allocation
* Better coordination of food preparation and dispatch
* Exception management for delayed orders
* Capacity planning using historical delivery patterns

A complete vehicle-routing optimization system was not implemented because the available dataset does not contain information such as depot locations, vehicle capacities, route networks, or real-time traffic data.

## 📁 Repository Structure

```text
Logistics-Data-Analytics/
│
├── data/
│   ├── Food_Delivery_Times.csv
│   └── Food_Delivery_Times_Cleaned_Week2.csv
│
├── reports/
│   ├── Week_1_Logistics_Strategic_Planning_Report.docx
│   ├── Week_2_Logistics_Data_Cleaning_Preprocessing_Report.docx
│   ├── Week_3_Logistics_EDA_Visualization_Report.docx
│   └── Week_4_Logistics_Predictive_Modeling_Optimization_Report.docx
│
├── results/
│   ├── Week_4_Model_Comparison.csv
│   └── Week_4_Cross_Validation_Results.csv
│
├── charts/
│   ├── week3/
│   └── week4/
│
└── README.md
```

## 🎯 Project Outcome

The project demonstrates an end-to-end logistics analytics workflow, from raw data preparation and exploratory analysis to predictive modeling and operational recommendations. The analysis shows that delivery distance, preparation time, traffic, weather, and other operational factors can be used to build a model capable of estimating delivery time and supporting logistics decision-making.

## 👨‍💻 Author

**Rishi Raj**

B.Tech Computer Science & Engineering

Raj Kumar Goel Institute of Technology (RKGIT), Ghaziabad
