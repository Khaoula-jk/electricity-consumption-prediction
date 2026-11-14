# Household Electricity Consumption Forecasting

**Key Skills Demonstrated:** Python, Pandas, Scikit-Learn, Data Cleaning, Feature Engineering, Time-Series Splitting, Leakage Prevention.

## Executive Summary
This project builds a Machine Learning model to forecast household electricity consumption using the UCI "Individual Household Electric Power Consumption" dataset (over 2 million minute-level records). 

The goal was not just to maximize accuracy, but to build a model that is **operationally useful** for grid forecasting. Through rigorous EDA and feature ablation, I identified and removed a data leakage feature, and deliberately rejected a "cheat" lag feature that artificially inflated accuracy but provided no real-world value.

## Data Cleaning & Feature Engineering
- **Parsing:** Handled raw text files with misaligned columns (8 fields instead of 9) by manually parsing lines and padding missing values.
- **Missing Values:** Dropped ~26,000 corrupted rows with aligned missingness across all meter columns.
- **Zeros:** Investigated zero values; determined they represented genuine appliance inactivity (e.g., Sub_metering_1 is 0 for 91% of the time) and retained them.
- **Leakage:** Dropped `Global_intensity` after finding a 1.00 correlation with the target (physical relation: P = V × I).
- **Features:** Extracted `hour`, `day_of_week`, and `month`, and used One-Hot Encoding so the linear model could learn non-linear daily cycles.

## Exploratory Data Analysis

**1. Daily Consumption Rhythm**
The plot below shows a clear bimodal pattern: consumption spikes in the morning (7-9 AM) and peaks significantly in the evening (6-10 PM). This proves that time-of-day is a highly relevant feature.
![Hourly Consumption](Khaoula-jk/electricity-consumption-prediction/main/hourly_consumption.png)

**2. Correlation & Leakage Detection**
The heatmap revealed that `Global_intensity` had a 1.00 correlation with the target. Because Intensity = Power / Voltage, including it would cause catastrophic data leakage. It was excluded.
![Correlation Heatmap](Khaoula-jk/electricity-consumption-prediction/main/correlation_heatmap.png)

## Model Evaluation & The "Lag Paradox"
I trained several Linear Regression models using a **chronological 80/20 train-test split** (no shuffling, to prevent future data leakage). 

| Model | Features Used | R² Score | MAE (kW) |
| :--- | :--- | :--- | :--- |
| **A: Baseline** | Time only (hour, day, month) | 0.1833 | 0.5887 |
| **B: Operational** | Time + `lag_1hour` + `lag_1day` | **0.2871** | **0.5261** |

### Why didn't I use a 1-minute lag?
In initial testing, adding a `lag_1min` feature skyrocketed the R² to **0.94**. However, I excluded this from the final model for two reasons:
1. **It's a cheat:** Minute-level power is highly autocorrelated. A 1-minute lag simply tells the model what the power *currently* is, not what it *will be*.
2. **It's useless in production:** Grid operators need to forecast hours or days ahead to manage supply. Predicting the next 60 seconds is not operationally valuable.

Model B (using 1-hour and 1-day lags) represents an honest forecasting model. The R² is lower, but it actually solves a real-world problem by providing a 1-hour lookahead forecast.

![Model Comparison](Khaoula-jk/electricity-consumption-prediction/main/model_comparison.png)

## How to Run
1. Clone this repo.
2. Download the raw dataset from [UCI](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) then create a data folder in the root directory and place `household_power_consumption.txt` inside it.
3. Run the Jupyter Notebook (`electricity_forecasting.ipynb`).