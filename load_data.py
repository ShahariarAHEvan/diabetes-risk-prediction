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