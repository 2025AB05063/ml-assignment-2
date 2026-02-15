Problem Statement
This project compares multiple classification models on the Breast Cancer Wisconsin dataset and evaluates their performance using standard evaluation metrics.

Dataset Description
Dataset: Breast Cancer Wisconsin Dataset
Instances: 569
Features: 30
Type: Binary Classification
Target: Malignant (1), Benign (0)

Models Used
Logistic Regression
Decision Tree
KNN
Naive Bayes
Random Forest
XGBoost

Model Performance Comparison
Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.96	0.99	0.95	0.97	0.96	0.92
Decision Tree	0.94	0.94	0.93	0.95	0.94	0.88
K	0.95	0.98	0.94	0.96	0.95	0.90
Naive Bayes	0.93	0.97	0.91	0.95	0.93	0.86
Random Forest	0.97	0.99	0.96	0.98	0.97	0.94
XGBoost	0.98	0.99	0.97	0.99	0.98	0.96

Observations
Logistic Regression performs well due to linear separability.
Decision Tree shows slight overfitting.
KNN performs well but sensitive to scaling.
Naive Bayes slightly lower accuracy due to independence assumption.
Random Forest improves stability and reduces overfitting.
XGBoost achieves best overall performance.