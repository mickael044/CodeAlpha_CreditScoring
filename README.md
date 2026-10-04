
# CodeAlpha Credit Scoring

Predicting whether a loan applicant is a good or bad credit risk using classical machine learning models.
Task 1 of the CodeAlpha Machine Learning Internship.

## Dataset
German Credit Data (UCI): 1000 samples, 20 features (financial history, employment, housing, etc.).
Loaded directly from the UCI Machine Learning Repository.

## Workflow
1. Categorical feature encoding (Label Encoding)
2. Exploratory data analysis
3. Stratified train/test split (80/20) and feature scaling
4. Models: Logistic Regression, Decision Tree, Random Forest (with `class_weight="balanced"`)
5. Hyperparameter tuning with GridSearchCV on XGBoost
6. Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix

## Results

| Stage | Best model | Accuracy | F1 | Recall (bad credit) | ROC-AUC |
|---|---|---|---|---|---|
| Balanced classes | Random Forest | 0.77 | 0.511 | 0.40 | 0.805 |
| Tuned XGBoost (GridSearchCV) | XGBoost | 0.76 | **0.619** | **0.65** | 0.80 |

## Key findings
- The dataset is imbalanced (700 good credit, 300 bad credit). Using `class_weight="balanced"` and `scale_pos_weight` nearly doubled recall on the minority class.
- Tuned XGBoost offers the best balance between precision and recall, catching 65% of bad credit cases.
- Most important features: checking account status, other installment plans, credit history, and savings account.
- In credit scoring, missing a bad credit case (false negative) is costlier than rejecting a good applicant, so recall on the bad credit class was prioritized.

## Tools
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn

## Notebook
[CodeAlpha_CreditScoring.ipynb](CodeAlpha_CreditScoring.ipynb)
