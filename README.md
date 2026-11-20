# tutorial.ipynb — Home Credit Default Risk

## Project overview

This notebook builds a Home Credit loan default risk model and generates probability predictions for the competition submission. Starting from a Logistic Regression validation ROC-AUC of **0.7334**, the work reaches **0.7580 ROC-AUC** with a weighted LightGBM and CatBoost prediction blend on a stratified 30% holdout set.

> The notebook reports ROC-AUC, not classification accuracy. The score above is the local validation result; no external leaderboard score is recorded in the notebook.

## What I built

- Explored the training and test data, inspected missing values, and visualized numeric and categorical feature distributions.
- Processed selected applicant, loan, bureau inquiry, demographic, and employment features.
- Engineered financial and affordability ratios, including credit-to-income, annuity-to-income, credit-to-goods, income and credit per family member, income after annualized annuity, estimated repayment duration, and a goods financing gap ratio.
- Combined the three external risk scores using their mean, product, and standard deviation; also created age and employment-to-age features.
- Handled missing values using medians, means, sentinel values, and zero filling; encoded categorical values and grouped/one-hot encoded car age.
- Compared Logistic Regression, Random Forest, LightGBM, and CatBoost using validation ROC-AUC.
- Used Matplotlib and Seaborn to plot feature distributions.
- Included Optuna search objectives for LightGBM and CatBoost. The search code is commented out in the notebook, so an executed Optuna result is not reported.
- Tested prediction blends. The best recorded validation result is a **60% LightGBM + 40% CatBoost** blend.
- Generated test-set probabilities and saved a submission CSV to `output/submission.csv` when the notebook is run.

## Validation results

| Model / blend | Validation ROC-AUC |
| --- | ---: |
| Logistic Regression | 0.7334 |
| Random Forest | 0.7377 |
| LightGBM | 0.7558 |
| CatBoost | 0.7559 |
| 80% LightGBM + 20% Random Forest | 0.7563 |
| **60% LightGBM + 40% CatBoost** | **0.7580** |

The increase from the Logistic Regression baseline to the best recorded blend is approximately **0.0246 ROC-AUC points** (about **2.46 percentage points** when expressed on a 0–100 scale).

## Repository contents

- `tutorial.ipynb` — data exploration, feature engineering, model comparison, blending, and submission generation.
- `input/train.csv`, `input/test.csv` — competition data expected by the notebook.
- `output/submission.csv` — generated predictions, created when the final notebook cells run.

## Running the notebook

The notebook was written for Google Colab and expects the input files under `input/`. It uses Python libraries including pandas, NumPy, Matplotlib, Seaborn, scikit-learn, LightGBM, CatBoost, and Optuna. Update the setup/path cells if running outside the original Colab/Drive environment, then run the notebook cells in order.

The submission output contains `SK_ID_CURR` and the predicted `TARGET` probability for each test row.
