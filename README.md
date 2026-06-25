from google.colab import files
uploaded = files.upload()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("credit_data.csv", sep=";")

if "customer_id" in df.columns:
    df.drop("customer_id", axis=1, inplace=True)

df["savings_income_ratio"] = df["total_savings_amount"] / (df["income"] + 1)
df["loan_income_ratio"] = df["total_active_loans_amount"] / (df["income"] + 1)
df["repayment_ratio"] = df["total_repaid_loans_amount"] / (df["total_active_loans_amount"] + 1)
df["family_size"] = df["number_of_children"] + 1

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("credit_worthy", axis=1)
y = df["credit_worthy"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

new_customer = pd.DataFrame([{
    "age": 35,
    "income": 70000,
    "employment_type": 1,
    "job_change_frequency": 1,
    "education_level": 1,
    "marital_status": 1,
    "number_of_children": 2,
    "owns_home": 1,
    "owns_vehicle": 1,
    "debt_history": 1,
    "overdue_payments": 0,
    "active_loans": 1,
    "avg_loan_term": 24,
    "total_active_loans_amount": 150000,
    "total_repaid_loans_amount": 300000,
    "has_savings": 1,
    "total_savings_amount": 200000,
    "has_other_financial_obligations": 0
}])

new_customer["savings_income_ratio"] = new_customer["total_savings_amount"] / (new_customer["income"] + 1)
new_customer["loan_income_ratio"] = new_customer["total_active_loans_amount"] / (new_customer["income"] + 1)
new_customer["repayment_ratio"] = new_customer["total_repaid_loans_amount"] / (new_customer["total_active_loans_amount"] + 1)
new_customer["family_size"] = new_customer["number_of_children"] + 1

new_customer = new_customer[X.columns]

prediction = model.predict(new_customer)[0]
probability = model.predict_proba(new_customer)[0][1]

print("\nPrediction:", prediction)
print("Probability:", round(probability * 100, 2), "%")

if prediction == 1:
    print("Customer is Creditworthy")
else:
    print("Customer is Not Creditworthy")
