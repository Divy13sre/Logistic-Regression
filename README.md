
1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_score, recall_score
)


2. Load and Prepare Dataset

   # Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 1 = malignant, 0 = benign

# Check class distribution
print(y.value_counts())

output:
1    357
0    212
Name: count, dtype: int64

3. Train/Test Split and Standardize

   # Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


4. Fit Logistic Regression Model

   model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict probabilities and labels
y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

5. Evaluation Metrics

   # Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

output:
Confusion Matrix:
 [[41  2]
 [ 1 70]]

Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.95      0.96        43
           1       0.97      0.99      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114

ROC-AUC Score: 0.99737962659679

![image](https://github.com/user-attachments/assets/242fcdb8-214d-4e1f-8e11-0d73c30bfde7)

6. Tune Threshold

   # Custom threshold
threshold = 0.4  # example threshold
y_pred_custom = (y_prob >= threshold).astype(int)

# Evaluate at custom threshold
print("Precision:", precision_score(y_test, y_pred_custom))
print("Recall:", recall_score(y_test, y_pred_custom))

output:

Precision: 0.9726027397260274
Recall: 1.0


7. Explanation of Sigmoid Function:

   The sigmoid function is used in logistic regression to map linear output to probability:
σ(z)= 1/1+e^-z
z=w⋅x+b

Output: probability between 0 and 1.

If probability ≥ threshold (default 0.5), classify as class 1; otherwise, class 0.




