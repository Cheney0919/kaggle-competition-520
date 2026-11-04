# AIPI 520 Kaggle Competition — Classical Music Meets Classical ML (Fall 2026)

Task: predict which previous patrons will buy a 2014–15 season subscription (AUROC).

Final model: logistic regression (public AUROC 0.9768).

## Run

1. Place competition data under `data/for_students/` (not included).
2. Run in order: `scripts/build_strict_features.py`, `scripts/build_clean_features.py`, `scripts/run_stage8.py`, `scripts/run_stage9.py`, `scripts/run_stage10.py`.

Requirements: Python, pandas, scikit-learn, xgboost.

## Not included

Raw competition data, derived feature tables, submission file, written report.
