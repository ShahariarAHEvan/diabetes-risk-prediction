\# Diabetes Risk Prediction — Model Comparison \& Bias-Variance Analysis



A machine learning project predicting diabetes risk from patient medical measurements, focused on comparing model complexity, demonstrating overfitting, and evaluating models appropriately for a medical context.



\## Problem



Given basic medical measurements (glucose, BMI, blood pressure, etc.), predict whether a patient is at risk of diabetes. This is a binary classification problem where the cost of errors is asymmetric: missing an actual diabetes case (false negative) is more dangerous than a false alarm (false positive), which shapes how the model should be evaluated.



\## Dataset



The Pima Indians Diabetes dataset (768 patient records, originally from the National Institute of Diabetes and Digestive and Kidney Diseases, accessed via a public mirror). Features include Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.



\*\*Data quality issue identified and handled:\*\* several columns (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI) used `0` as a placeholder for missing data — biologically impossible values (e.g., zero blood glucose). Up to 44% of values were missing in some columns (Insulin). Rather than dropping these rows (which would discard nearly half the dataset), missing values were imputed using the column median, which is robust to outliers.



\## Methodology



1\. \*\*Data cleaning\*\* — identified and imputed missing values disguised as zeros

2\. \*\*Train/test split\*\* — 80/20 split, with a fixed random seed for reproducibility

3\. \*\*Baseline model\*\* — Logistic Regression

4\. \*\*Complex model\*\* — Random Forest (default parameters)

5\. \*\*Regularized model\*\* — Random Forest with constrained tree depth and minimum leaf size

6\. \*\*Evaluation\*\* — accuracy, confusion matrix, precision/recall, with attention to the medical cost of false negatives



\## Results



| Model | Training Accuracy | Testing Accuracy | Train-Test Gap |

|---|---|---|---|

| Logistic Regression | 76.9% | 75.3% | 1.6% |

| Random Forest (default) | 100% | 74.7% | 25.3% |

| Random Forest (tuned) | 83.4% | 77.9% | 5.5% |



The default Random Forest achieved perfect training accuracy but performed no better than the baseline on test data — a clear case of overfitting driven by excessive model complexity (high variance, low bias). Constraining the model (`max\_depth=5`, `min\_samples\_leaf=5`) reduced training accuracy but improved test accuracy and shrank the train-test gap by nearly 5x, demonstrating the bias-variance tradeoff directly.



\*\*On the tuned model specifically:\*\*

\- Recall for the diabetic class: 62% — the model misses roughly 1 in 3 actual diabetes cases

\- Precision for the diabetic class: 72%



This gap between overall accuracy (77.9%) and diabetic-class recall (62%) is the key finding: \*\*accuracy alone is a misleading metric for this problem.\*\* In a real clinical deployment, this model would need further tuning (e.g., adjusting the classification threshold, or using class weighting) to reduce false negatives, even at the cost of overall accuracy.



\## Tech Stack



\- Python, pandas, NumPy

\- scikit-learn (Logistic Regression, Random Forest, evaluation metrics)



\## What I'd Improve Next



\- Address class imbalance (500 non-diabetic vs. 268 diabetic) using class weighting or resampling

\- Tune the classification threshold specifically to improve recall for the diabetic class

\- Try feature engineering (e.g., BMI categories,

