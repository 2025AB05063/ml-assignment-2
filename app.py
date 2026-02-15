import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

st.title("ML Assignment 2 - Classification Models Comparison")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_option = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

if model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=5000)
elif model_option == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_option == "KNN":
    model = KNeighborsClassifier()
elif model_option == "Naive Bayes":
    model = GaussianNB()
elif model_option == "Random Forest":
    model = RandomForestClassifier()
else:
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

st.subheader("Evaluation Metrics")
st.write("Accuracy:", accuracy)
st.write("AUC:", auc)
st.write("Precision:", precision)
st.write("Recall:", recall)
st.write("F1 Score:", f1)
st.write("MCC:", mcc)

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# CSV Upload
st.subheader("Upload Test Dataset (Optional)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    uploaded_data = pd.read_csv(uploaded_file)
    st.write(uploaded_data.head())
