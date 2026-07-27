import pandas as pd

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(url, names=column_names)

print(df.shape)
print(df.head())

print((df == 0).sum())


import numpy as np

columns_with_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in columns_with_missing:
    df[col] = df[col].replace(0, np.nan)
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

print("\nAfter cleaning:")
print((df == 0).sum())

from sklearn.model_selection import train_test_split

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"\nLogistic Regression:")
print(f"Training accuracy: {train_accuracy:.3f}")
print(f"Testing accuracy: {test_accuracy:.3f}")

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_train_accuracy = rf_model.score(X_train, y_train)
rf_test_accuracy = rf_model.score(X_test, y_test)

print(f"\nRandom Forest:")
print(f"Training accuracy: {rf_train_accuracy:.3f}")
print(f"Testing accuracy: {rf_test_accuracy:.3f}")

rf_model_tuned = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)
rf_model_tuned.fit(X_train, y_train)

rf_tuned_train_accuracy = rf_model_tuned.score(X_train, y_train)
rf_tuned_test_accuracy = rf_model_tuned.score(X_test, y_test)

print(f"\nRandom Forest (tuned):")
print(f"Training accuracy: {rf_tuned_train_accuracy:.3f}")
print(f"Testing accuracy: {rf_tuned_test_accuracy:.3f}")

from sklearn.metrics import classification_report, confusion_matrix

y_pred = rf_model_tuned.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))