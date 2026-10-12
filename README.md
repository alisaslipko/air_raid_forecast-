# Air Raid Alert Forecasting in Ukraine

## Formalized Problem Statement

This project solves a **binary classification problem on time-series data** for forecasting.

* **Objective:** Predict the fact/probability of an air raid alert occurring in a specific Ukrainian region within a predefined future time window (e.g., `target_60_120m` — an alert occurring between 60 and 120 minutes from the current moment $t$).
* **Key Challenge & Baseline:** Air raid status has high inertia (if there is no alert now, it is highly probable there won't be one in an hour). Therefore, the primary challenge is to predict state transitions. To prove predictive value, the ML model must reliably outperform a Naive Baseline (a persistence model that assumes the future state equals the current state $is\_alert_t$).

*  **Link to the data**: https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/refs/heads/main/datasets/official_data_en.csv from the github https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset
