from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from difflib import get_close_matches
import os
import json
import random

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

class MedicalAISystem:
    def __init__(self):
        self.symptoms_list = []
        self.diseases = []
        self.model = None
        self.mlb = None
        self.disease_info = {}
        self.load_kaggle_dataset()
        
    def load_kaggle_dataset(self):
        """Load and preprocess the Kaggle dataset"""
        try:
            # Try to load the Kaggle dataset
            # Adjust the filename based on your actual dataset
            kaggle_files = [
                "dataset.csv", 
                "medical_symptoms.csv",
                "disease_symptoms.csv",
                "Disease_symptom_dataset.csv"
            ]
            
            df = None
            for file in kaggle_files:
                if os.path.exists(file):
                    df = pd.read_csv(file)
                    print(f"Loaded dataset from {file}")
                    break
            
            if df is None:
                print("No Kaggle dataset found. Using sample data.")
                self.load_sample_data()
                return
            
            # Process the dataset
            # Assuming the dataset has columns: 'Disease', 'Symptom_1', 'Symptom_2', ...
            symptom_columns = [col for col in df.columns if 'Symptom' in col or 'symptom' in col]
            
            if not symptom_columns:
                # Try to infer structure
                if len(df.columns) >= 2:
                    # Assume first column is disease, rest are symptoms
                    disease_col = df.columns[0]
                    symptom_columns = list(df.columns[1:])
                else:
                    raise ValueError("Could not identify symptom columns")
            
            # Extract all unique symptoms
            all_symptoms = set()
            for col in symptom_columns:
                unique_symptoms = df[col].dropna().unique()
                all_symptoms.update([str(s).strip().lower() for s in unique_symptoms])
            
            self.symptoms_list = sorted(list(all_symptoms))
            self.diseases = df.iloc[:, 0].unique().tolist()
            
            # Build disease info dictionary
            for _, row in df.iterrows():
                disease = str(row.iloc[0])
                symptoms = []
                for col in symptom_columns:
                    symptom = row[col]
                    if pd.notna(symptom):
                        symptoms.append(str(symptom).strip().lower())
                
                if disease not in self.disease_info:
                    self.disease_info[disease] = {
                        'symptoms': symptoms,
                        'description': self.get_disease_description(disease),
                        'treatment': self.get_default_treatment(disease)
                    }
            
            print(f"Loaded {len(self.diseases)} diseases and {len(self.symptoms_list)} symptoms")
            
            # Train ML model
            self.train_model(df, symptom_columns)
            
        except Exception as e:
            print(f"Error loading Kaggle dataset: {e}")
            self.load_sample_data()
    
    def load_sample_data(self):
        """Load sample data if Kaggle dataset not available"""
        self.symptoms_list = [
            'fever', 'cough', 'fatigue', 'headache', 'nausea',
            'vomiting', 'diarrhea', 'rash', 'itching', 'chest pain',
            'shortness of breath', 'dizziness', 'sore throat',
            'runny nose', 'sneezing', 'body aches', 'chills',
            'muscle pain', 'loss of taste', 'loss of smell'
        ]
        
        self.diseases = ['Influenza', 'Common Cold', 'COVID-19', 'Bronchitis', 
                        'Stomach Flu', 'Allergic Reaction', 'Migraine']
        
        # Create sample disease info
        for disease in self.diseases:
            self.disease_info[disease] = {
                'symptoms': random.sample(self.symptoms_list, random.randint(3, 6)),
                'description': f"Description for {disease}",
                'treatment': [
                    "Rest and hydration",
                    "Over-the-counter medication as needed",
                    "Monitor symptoms"
                ]
            }
        
        # Train a simple model
        self.train_sample_model()
    
    def train_model(self, df, symptom_columns):
        """Train ML model on the dataset"""
        try:
            # Prepare training data
            X = []
            y = []
            
            for _, row in df.iterrows():
                disease = str(row.iloc[0])
                symptoms = []
                for col in symptom_columns:
                    symptom = row[col]
                    if pd.notna(symptom):
                        symptoms.append(str(symptom).strip().lower())
                
                if symptoms:
                    X.append(symptoms)
                    y.append(disease)
            
            # Use MultiLabelBinarizer
            self.mlb = MultiLabelBinarizer(classes=self.symptoms_list)
            X_encoded = self.mlb.fit_transform(X)
            
            # Train Random Forest
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_encoded, y)
            
            print("ML model trained successfully")
            
            # Save model
            joblib.dump(self.model, 'medical_model.joblib')
            joblib.dump(self.mlb, 'symptoms_binarizer.joblib')
            
        except Exception as e:
            print(f"Error training model: {e}")
            self.model = None
    
    def train_sample_model(self):
        """Train a simple model on sample data"""
        # Create synthetic training data
        X = []
        y = []
        
        for disease, info in self.disease_info.items():
            symptoms = info['symptoms']
            X.append(symptoms)
            y.append(disease)
        
        self.mlb = MultiLabelBinarizer()
        X_encoded = self.mlb.fit_transform(X)
        
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X_encoded, y)
    
    def get_disease_description(self, disease):
        """Get description for a disease"""
        descriptions = {
            'Influenza': 'Viral infection of the respiratory system causing fever, cough, and body aches.',
            'Common Cold': 'Mild viral infection of the nose and throat.',
            'COVID-19': 'Respiratory illness caused by the SARS-CoV-2 virus.',
            'Bronchitis': 'Inflammation of the bronchial tubes in the lungs.',
            'Migraine': 'Neurological condition characterized by severe headaches.',
            'Allergic Reaction': 'Immune system response to a foreign substance.',
            'Stomach Flu': 'Viral infection causing inflammation of the stomach and intestines.'
        }
        return descriptions.get(disease, f"Information about {disease}")
    
    def get_default_treatment(self, disease):
        """Get default treatment plan for a disease"""
        treatments = {
            'Influenza': [
                'Rest and stay hydrated',
                'Take antiviral medication if prescribed',
                'Use over-the-counter fever reducers',
                'Monitor for complications'
            ],
            'Common Cold': [
                'Get plenty of rest',
                'Drink warm liquids',
                'Use saline nasal spray',
                'Take over-the-counter cold medicine'
            ],
            'COVID-19': [
                'Isolate for at least 5 days',
                'Monitor oxygen levels if available',
                'Rest and stay hydrated',
                'Seek medical attention if breathing difficulties occur'
            ],
            'Bronchitis': [
                'Use a humidifier',
                'Drink plenty of fluids',
                'Avoid irritants like smoke',
                'Take cough medicine as needed'
            ]
        }
        return treatments.get(disease, [
            'Consult with a healthcare professional',
            'Rest and monitor symptoms',
            'Stay hydrated',
            'Follow any prescribed medications'
        ])
    
    def predict_disease(self, symptoms):
        """Predict disease based on symptoms"""
        if not self.model:
            return self.rule_based_predict(symptoms)
        
        try:
            # Encode symptoms
            symptoms_lower = [s.lower() for s in symptoms]
            symptoms_encoded = self.mlb.transform([symptoms_lower])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(symptoms_encoded)[0]
            disease_probs = list(zip(self.model.classes_, probabilities))
            disease_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Get top prediction
            top_disease, confidence = disease_probs[0]
            
            # Calculate confidence score (0-100)
            confidence_score = confidence * 100
            
            # Get disease info
            disease_info = self.disease_info.get(top_disease, {})
            
            # Check for emergency conditions
            emergency_conditions = ['chest pain', 'shortness of breath', 'severe headache', 
                                   'difficulty breathing', 'loss of consciousness']
            has_emergency = any(symptom in emergency_conditions for symptom in symptoms_lower)
            
            return {
                'disease': top_disease,
                'confidence': round(confidence_score, 1),
                'description': disease_info.get('description', ''),
                'treatment': disease_info.get('treatment', []),
                'emergency': has_emergency,
                'emergency_message': 'Some of your symptoms require immediate medical attention. Please go to the nearest emergency room or call emergency services.' if has_emergency else None,
                'self_care': [
                    'Get adequate rest',
                    'Stay hydrated',
                    'Monitor your symptoms',
                    'Avoid strenuous activities'
                ]
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.rule_based_predict(symptoms)
    
    def rule_based_predict(self, symptoms):
        """Fallback to rule-based prediction if ML fails"""
        symptoms_lower = [s.lower() for s in symptoms]
        
        best_match = None
        best_score = 0
        
        for disease, info in self.disease_info.items():
            disease_symptoms = info.get('symptoms', [])
            match_count = len(set(symptoms_lower) & set(disease_symptoms))
            total_symptoms = len(disease_symptoms)
            
            if total_symptoms > 0:
                score = match_count / total_symptoms
                if score > best_score:
                    best_score = score
                    best_match = disease
        
        confidence = best_score * 100 if best_match else 50
        
        return {
            'disease': best_match or 'Unable to determine specific condition',
            'confidence': round(confidence, 1),
            'description': self.disease_info.get(best_match, {}).get('description', 'Consult a healthcare professional for accurate diagnosis.'),
            'treatment': self.disease_info.get(best_match, {}).get('treatment', [
                'Consult with a healthcare professional',
                'Rest and monitor symptoms'
            ]),
            'emergency': any(s in ['chest pain', 'shortness of breath'] for s in symptoms_lower),
            'self_care': ['Get rest', 'Stay hydrated', 'Monitor symptoms']
        }
    
    def get_symptom_suggestions(self, selected_symptoms):
        """Get suggested symptoms based on current selection"""
        suggestions = set()
        
        for disease, info in self.disease_info.items():
            disease_symptoms = set(info.get('symptoms', []))
            selected_set = set([s.lower() for s in selected_symptoms])
            
            # If we have some matching symptoms
            if selected_set & disease_symptoms:
                # Suggest symptoms from this disease that aren't already selected
                suggestions.update(disease_symptoms - selected_set)
        
        # Return top 5 suggestions
        return list(suggestions)[:5]

# Initialize the AI system
ai_system = MedicalAISystem()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/symptoms')
def get_symptoms():
    """Return list of symptoms"""
    return jsonify({
        'symptoms': ai_system.symptoms_list,
        'count': len(ai_system.symptoms_list)
    })

@app.route('/diagnose', methods=['POST'])
def diagnose():
    """Process symptoms and return diagnosis"""
    try:
        data = request.json
        
        if not data or 'symptoms' not in data:
            return jsonify({
                'success': False,
                'error': 'No symptoms provided'
            }), 400
        
        symptoms = data['symptoms']
        
        print(f"Diagnosing for symptoms: {symptoms}")
        
        # Get diagnosis from AI system
        result = ai_system.predict_disease(symptoms)
        
        # Add additional info
        result['success'] = True
        result['symptoms_analyzed'] = symptoms
        result['model_used'] = 'NASA HUNCH Medical AI v1.0'
        
        # Add severity analysis
        symptom_details = data.get('symptom_details', {})
        if symptom_details:
            avg_severity = np.mean([details.get('severity', 5) for details in symptom_details.values()])
            result['avg_symptom_severity'] = round(avg_severity, 1)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Diagnosis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'disease': 'Diagnosis Unavailable',
            'confidence': 0,
            'treatment': ['Please consult a healthcare professional directly.']
        }), 500

@app.route('/suggest', methods=['POST'])
def suggest_symptoms():
    """Get suggested symptoms based on current selection"""
    try:
        data = request.json
        selected = data.get('symptoms', [])
        
        suggestions = ai_system.get_symptom_suggestions(selected)
        
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'suggestions': []
        }), 500

@app.route('/emergency_check', methods=['POST'])
def emergency_check():
    """Check if symptoms indicate emergency"""
    try:
        data = request.json
        symptoms = data.get('symptoms', [])
        
        emergency_keywords = [
            'chest pain', 'shortness of breath', 'difficulty breathing',
            'severe pain', 'unconscious', 'severe headache',
            'sudden weakness', 'paralysis', 'severe bleeding',
            'thoughts of self-harm', 'suicidal'
        ]
        
        has_emergency = any(
            any(keyword in symptom.lower() for keyword in emergency_keywords)
            for symptom in symptoms
        )
        
        return jsonify({
            'emergency': has_emergency,
            'message': 'Seek immediate medical attention!' if has_emergency else 'No immediate emergency detected.'
        })
        
    except Exception as e:
        return jsonify({
            'emergency': True,
            'message': 'Error in assessment - when in doubt, seek medical help.'
        }), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'diseases_loaded': len(ai_system.diseases),
        'symptoms_loaded': len(ai_system.symptoms_list),
        'model_ready': ai_system.model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
    
    
# from flask import Flask, render_template, request, jsonify
# import os
# import sys

# # Manually import all modules
# project_root = os.path.dirname(os.path.abspath(__file__))

# # Add to path
# sys.path.insert(0, project_root)
# sys.path.insert(0, os.path.join(project_root, "User"))
# sys.path.insert(0, os.path.join(project_root, "Machine_Learning"))

# # Import using absolute paths
# import importlib.util

# def import_module_from_file(module_name, file_path):
#     spec = importlib.util.spec_from_file_location(module_name, file_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module

# # Import all modules
# MachineLearningEngine = import_module_from_file(
#     "MachineLearningEngine",
#     os.path.join(project_root, "Machine_Learning", "MachineLearningEngine.py")
# ).MachineLearningEngine

# InferenceEngine = import_module_from_file(
#     "InferenceEngine",
#     os.path.join(project_root, "User", "InferenceEngine.py")
# ).InferenceEngine

# KnowledgeBase = import_module_from_file(
#     "KnowledgeBase",
#     os.path.join(project_root, "User", "KnowledgeBase.py")
# ).KnowledgeBase

# OutputModule = import_module_from_file(
#     "OutputModule",
#     os.path.join(project_root, "User", "OutputModule.py")
# ).OutputModule

# SymptomCollector = import_module_from_file(
#     "SymptomCollector",
#     os.path.join(project_root, "User", "SymptomCollector.py")
# ).SymptomCollector

# # Now import DiagnosisEngine
# DiagnosisEngine = import_module_from_file(
#     "DiagnosisEngine",
#     os.path.join(project_root, "User", "DiagnosisEngine_new.py")
# ).DiagnosisEngine

# app = Flask(__name__)
# engine = DiagnosisEngine()

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/diagnose', methods=['POST'])
# def diagnose():
#     try:
#         data = request.json
#         symptoms = data.get('symptoms', [])
#         symptom_details = data.get('symptom_details', {})
        
#         if not symptoms:
#             return jsonify({'error': 'No symptoms provided'}), 400
        
#         # Pass symptom details if available
#         disease, confidence = engine.diagnose(symptoms)
        
#         # If we have duration/severity, we could use them to adjust confidence
#         if symptom_details:
#             # You could add logic here to adjust confidence based on duration/severity
#             print(f"Symptom details received: {symptom_details}")
        
#         treatment = engine.suggest_treatment_plan()
#         report = engine.generate_report()
        
#         return jsonify({
#             'success': True,
#             'diagnosis': disease,
#             'confidence': round(confidence * 100, 2),
#             'treatment': treatment,
#             'report': report,
#             'symptom_details': symptom_details  # Include in response
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/symptoms', methods=['GET'])
# def get_symptoms():
#     symptoms = [
#         "Fever", "Cough", "Headache", "Fatigue", 
#         "Nausea", "Shortness of Breath", "Sore Throat", 
#         "Muscle Pain", "Loss of Taste", "Chest Pain",
#         "Mucus", "Chills", "Runny Nose", "Body Aches",
#         "Vomiting", "Diarrhea", "Rash", "Itching",
#         "Dizziness", "Blurred Vision", "Wheezing",
#         "Loss of Smell", "Sneezing", "Chest Tightness"
#     ]
#     return jsonify({'symptoms': symptoms})

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)