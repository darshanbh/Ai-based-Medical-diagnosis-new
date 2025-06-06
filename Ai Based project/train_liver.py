import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
df = pd.read_csv("indian_liver_patient.csv")
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df = df.dropna()

# Rename columns
df = df.rename(columns={
    'Name':'name',
    'Age': 'age',
    'Total_Bilirubin': 'total_bilirubin',
    'Direct_Bilirubin': 'direct_bilirubin',
    'Alkaline_Phosphotase': 'alk_phosphotase',
    'Alamine_Aminotransferase': 'alamine_aminotransferase',
    'Aspartate_Aminotransferase': 'aspartate_aminotransferase',
    'Total_Proteins': 'total_protiens',
    'Albumin': 'albumin',
    'Albumin_and_Globulin_Ratio': 'albumin_globulin_ratio',
})

df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

X = df.drop(['Dataset'], axis=1)
y = df['Dataset']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/liver_model.pkl")
print("✅ Liver model trained successfully.")
