import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_diabetes_model(data):
    # Rename columns (optional, to match frontend names)
    data = data.rename(columns={
        'Pregnancies': 'pregnancies',
        'Glucose': 'glucose',
        'BloodPressure': 'blood_pressure',
        'SkinThickness': 'skin_thickness',
        'Insulin': 'insulin',
        'BMI': 'bmi',
        'DiabetesPedigreeFunction': 'dpf',
        'Age': 'age'
    })

    # Drop rows with critical zero values
    cols_to_check = ['glucose', 'blood_pressure', 'bmi']
    data = data[(data[cols_to_check] != 0).all(axis=1)]

    # Features and target
    X = data.drop(['Outcome'], axis=1)
    y = data['Outcome']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/diabetes_model.pkl')

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Diabetes Model Accuracy: {accuracy:.4f}")

# Load dataset and train
if __name__ == "__main__":
    data = pd.read_csv('diabetes.csv')  # Make sure this file exists in the same folder
    train_diabetes_model(data)
