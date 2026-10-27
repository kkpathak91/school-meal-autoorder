# Constraint-Aware Auto-Ordering for School Meal Subscriptions

Research project: *Learning When to Repeat and When to Explore*. It predicts which entree a student would choose
on each meal-plan day, subject to the day's school menu, allergens and the subscription plan.

## Repository structure
```
src/generate_school_meal_data.py   synthetic data generator (seed 42 -> identical data)
src/validate_dataset.py            hard-rule checks + calibration report
docs/DATA_CARD.md                  data card: tables, joins, assumptions, limitations
docs/VALIDATION_REPORT.txt         validation output for the published dataset
docs/Synopsis_*.pdf                research synopsis
notebooks/                         analysis notebooks (added weekly)
data/                              dataset location (CSV files are not committed)
```

## Get the data (two options)
1. **Download** `school_meal_synthetic_v2.zip` from the Releases page (v1.0) and unzip it into `data/raw/`.
2. **Regenerate** it (about 2 minutes, needs about 3 GB of RAM):
```bash
git clone https://github.com/kkpathak91/school-meal-autoorder.git
cd school-meal-autoorder
pip install -r requirements.txt
python src/generate_school_meal_data.py --seed 42 --students 30000 --out ./data/raw
python src/validate_dataset.py ./data/raw  # writes data/raw/VALIDATION_REPORT.txt
```

## Key facts
- 30,000 students, 105 schools, Aug 2023 - Jun 2025, about 2.7M lunch records. Fully synthetic and anonymised.
- Live lunches are those with an empty `LUNCHCANCELED`. Exclude `LUNCHTYPE = IOU` from modelling.
- `data/_ground_truth/` is for evaluation only. Never use it as model input.

## Research questions
RQ1 repeat vs explore | RQ2 menu-change robustness | RQ3 cold start | RQ4 auto-order acceptance

## Notebooks
1. `notebooks/02_eda_by_hypothesis.ipynb`: baseline EDA for hypotheses H1–H4
2. `notebooks/03_EDA_Features_Clustering.ipynb`: data cleaning audit, raw EDA, feature engineering, k-means segmentation, preliminary models (RQ1, RQ2, RQ4)

Run them from inside `notebooks/` after placing the data in `data/raw/`. All random processes use seed 42.

## Processed data
`data/processed/student_features_clusters.csv`: 21,668 students × 11 engineered features + segment label.
