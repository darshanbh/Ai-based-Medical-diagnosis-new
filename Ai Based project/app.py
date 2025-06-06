from flask import Flask, render_template, request
import sqlite3
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score

app = Flask(__name__)

def load_datasets():
    liver_data = pd.read_csv('indian_liver_patient.csv')
    liver_data['Gender'] = liver_data['Gender'].map({'Male': 1, 'Female': 0})
    liver_data = liver_data.dropna()
    
    diabetes_data = pd.read_csv('diabetes.csv')
    heart_data = pd.read_csv('heart.csv')
    
    if not os.path.exists('models/liver_model.pkl'):
        train_liver_model(liver_data)
    if not os.path.exists('models/diabetes_model.pkl'):
        train_diabetes_model(diabetes_data)
    if not os.path.exists('models/heart.pkl'):
        train_heart_model(heart_data)

def train_liver_model(data):
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data = data.dropna()
    data = data.rename(columns={
        'Age': 'age',
        'Total_Bilirubin': 'total_bilirubin',
        'Direct_Bilirubin': 'direct_bilirubin',
        'Alkaline_Phosphotase': 'alk_phosphotase',
        'Alamine_Aminotransferase': 'alamine_aminotransferase',
        'Aspartate_Aminotransferase': 'aspartate_aminotransferase',
        'Total_Proteins': 'total_protiens',
        'Albumin': 'albumin',
        'Albumin_and_Globulin_Ratio': 'albumin_globulin_ratio'
    })
    data['Dataset'] = data['Dataset'].map({1: 1, 2: 0})
    X = data.drop(['Dataset'], axis=1)
    y = data['Dataset']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/liver_model.pkl')

def train_diabetes_model(data):
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
    cols_to_check = ['glucose', 'blood_pressure', 'bmi']
    data = data[(data[cols_to_check] != 0).all(axis=1)]
    X = data.drop(['Outcome'], axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/diabetes_model.pkl')

def train_heart_model(data):
    data = data.rename(columns={
        'age':'age', 'sex':'sex', 'cp': 'chest_pain_type',
        'trestbps': 'resting_bp', 'chol': 'cholesterol',
        'fbs': 'fasting_bs', 'restecg': 'rest_ecg', 'thalach': 'max_heart_rate',
        'exang': 'exercise_angina', 'oldpeak': 'st_depression',
        'slope': 'slope_peak_ex', 'ca': 'num_major_vessels',
        'thal': 'thalassemia'
    })
    cols_to_check = ['cholesterol', 'resting_bp', 'max_heart_rate']
    data = data[(data[cols_to_check] != 0).all(axis=1)]
    X = data.drop(['target'], axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/heart.pkl')

def init_db():
    if not os.path.exists('diagnosis.db'):
        conn = sqlite3.connect('diagnosis.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE records
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT, age INTEGER, gender TEXT, 
                      disease_type TEXT, symptoms TEXT, 
                      diagnosis TEXT, probability REAL)''')
        conn.commit()
        conn.close()

init_db()
load_datasets()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<disease_type>', methods=['GET'])
def predict(disease_type):
    return render_template('diagnosis_form.html', disease_type=disease_type)

@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        print("Form data received:", request.form)

        disease_type = request.form.get('disease_type')
        if not disease_type:
            return "Missing disease_type field in the form.", 400

        name = request.form.get('name', 'Anonymous')
        age = int(request.form.get('age', 0))
        gender = request.form.get('gender', 'male').lower()

        if disease_type == 'liver':
            input_data = [[
                age,
                1 if gender == 'male' else 0,
                float(request.form['total_bilirubin']),
                float(request.form['direct_bilirubin']),
                int(request.form['alk_phosphotase']),
                int(request.form['alamine_aminotransferase']),
                int(request.form['aspartate_aminotransferase']),
                float(request.form['total_protiens']),
                float(request.form['albumin']),
                float(request.form['albumin_globulin_ratio'])
            ]]
            model = joblib.load('models/liver_model.pkl')
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            diagnosis = "Liver Disease" if prediction == 1 else "No Liver Disease"

        elif disease_type == 'diabetes':
            input_data = [[
                int(request.form['pregnancies']),
                int(request.form['glucose']),
                int(request.form['blood_pressure']),
                int(request.form['skin_thickness']),
                int(request.form['insulin']),
                float(request.form['bmi']),
                float(request.form['dpf']),
                age
            ]]
            model = joblib.load('models/diabetes_model.pkl')
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            diagnosis = "Diabetes" if prediction == 1 else "No Diabetes"

        elif disease_type == 'heart':
            input_data = [[
                age,
                1 if request.form['sex'].lower() == 'male' or request.form['sex'] == '1' else 0,
                int(request.form['cp']),
                int(request.form['trestbps']),
                int(request.form['chol']),
                int(request.form['fbs']),
                int(request.form['restecg']),
                int(request.form['thalach']),
                int(request.form['exang']),
                float(request.form['oldpeak']),
                int(request.form['slope']),
                int(request.form['ca']),
                int(request.form['thal'])
            ]]
            model = joblib.load('models/heart.pkl')
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            diagnosis = "Heart Disease" if prediction == 1 else "No Heart Disease"

        else:
            return "Invalid disease type selected.", 400

        # Store in DB
        conn = sqlite3.connect('diagnosis.db')
        c = conn.cursor()
        c.execute("""INSERT INTO records 
                     (name, age, gender, disease_type, symptoms, diagnosis, probability) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (name, age, gender, disease_type, str(dict(request.form)), diagnosis, float(probability)))
        conn.commit()
        conn.close()

        return render_template("result.html",
                               name=name,
                               diagnosis=diagnosis,
                               probability=round(probability*100, 2),
                               disease_type=disease_type)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
