import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, log_loss, accuracy_score, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv("../../datasets/loan_data.csv")
X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

num_features = ["Age", "Income", "LoanAmount", "CreditScore", "YearsAtJob"]
cat_features = ["Education", "MaritalStatus", "HasMortgage"]
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(), cat_features)
    ]
)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])

# Обучение
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
log_loss = log_loss(y_test, pipeline.predict_proba(X_test))
accuracy = accuracy_score(y_test, y_pred)

print(f"Log loss: {log_loss:.4f}; accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

new_client = pd.DataFrame({
    "Age": [22],
    "Income": [30000],
    "LoanAmount": [12000],
    "CreditScore": [580],
    "YearsAtJob": [1],
    "Education": ["Bachelor"],
    "MaritalStatus": ["Single"],
    "HasMortgage": ["No"]
})
prediction = pipeline.predict(new_client)
probability = pipeline.predict_proba(new_client)
print(prediction)
print(probability)
