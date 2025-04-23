# 🎧 Predict Podcast Listening Time - Kaggle Playground Series S5E4

This repository contains my submission for the [Kaggle competition: Playground Series - Season 5, Episode 4](https://www.kaggle.com/competitions/playground-series-s5e4), where the goal was to predict how long users will listen to a given podcast episode.

---

## 📌 Problem Statement

Given a dataset of podcast metadata (e.g., genre, publication time, host and guest popularity, sentiment, etc.), the task is to predict the expected **listening time** in minutes for each episode. This is a regression problem with potential nonlinear relationships and missing data challenges.

---

## 🛠️ Approaches

### ✅ Code 1: Quantile CatBoost with Log-Transformed Target

- Uses **log1p transformation** on target to stabilize variance.
- Extracts **feature interactions** and engineered metrics like:
  - `Ad_Density` = Number of Ads / Episode Length
  - `HostGuest_Mult` = Host × Guest Popularity
- CatBoost with `Quantile:alpha=0.5` for robust regression (median).
- Light categorical encoding and missing value imputation.

📉 **Validation Metrics**  
`RMSE: 13.19`  
`MAE : 9.52`

📄 Output: `submission.csv`

---

### ✅ Code 2: Hybrid Aggregation + CatBoost

- Adds **aggregated statistics** at the Podcast, Genre, and Day levels.
- Label encodes all categorical variables.
- Uses median/mode imputation for numerical and categorical features.
- No log transformation on target.

📉 **Validation Metrics**  
`RMSE: 13.12`  
`MAE : 9.47`

📄 Output: `submission_hybrid_catboost.csv`

---

## 📊 Evaluation Metric

Submissions are evaluated using **Root Mean Squared Error (RMSE)**.

---

## 🚀 Getting Started

1. Clone the repo
2. Install dependencies (e.g., `catboost`, `pandas`, `scikit-learn`)
3. Place `train.csv` and `test.csv` in the `data/` directory
4. Run the models from the `code/` folder:

```bash
python code/model_1_quantile_catboost.py
# or
python code/model_2_hybrid_features_catboost.py

```

---

## 📈 Submission Results

| Model                        | RMSE  | MAE  |
|-----------------------------|-------|------|
| Quantile CatBoost (Log Y)   | 13.19 | 9.52 |
| Hybrid CatBoost (Raw Y)     | 13.12 | 9.47 |

---

## 📚 Acknowledgements

- [Kaggle - Playground Series S5E4](https://www.kaggle.com/competitions/playground-series-s5e4)

---

## 🧠 Author

This project was built and submitted by **Shivam A**.  
Feel free to reach out or contribute!
