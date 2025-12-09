from flask import Flask, render_template, request, jsonify
import os
import sys

# Manually import all modules
project_root = os.path.dirname(os.path.abspath(__file__))

# Add to path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "User"))
sys.path.insert(0, os.path.join(project_root, "Machine_Learning"))

# Import using absolute paths
import importlib.util

def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import all modules
MachineLearningEngine = import_module_from_file(
    "MachineLearningEngine",
    os.path.join(project_root, "Machine_Learning", "MachineLearningEngine.py")
).MachineLearningEngine

InferenceEngine = import_module_from_file(
    "InferenceEngine",
    os.path.join(project_root, "User", "InferenceEngine.py")
).InferenceEngine

KnowledgeBase = import_module_from_file(
    "KnowledgeBase",
    os.path.join(project_root, "User", "KnowledgeBase.py")
).KnowledgeBase

OutputModule = import_module_from_file(
    "OutputModule",
    os.path.join(project_root, "User", "OutputModule.py")
).OutputModule

SymptomCollector = import_module_from_file(
    "SymptomCollector",
    os.path.join(project_root, "User", "SymptomCollector.py")
).SymptomCollector

# Now import DiagnosisEngine
DiagnosisEngine = import_module_from_file(
    "DiagnosisEngine",
    os.path.join(project_root, "User", "DiagnosisEngine_new.py")
).DiagnosisEngine

app = Flask(__name__)
engine = DiagnosisEngine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        data = request.json
        symptoms = data.get('symptoms', [])
        symptom_details = data.get('symptom_details', {})
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        # Pass symptom details if available
        disease, confidence = engine.diagnose(symptoms)
        
        # If we have duration/severity, we could use them to adjust confidence
        if symptom_details:
            # You could add logic here to adjust confidence based on duration/severity
            print(f"Symptom details received: {symptom_details}")
        
        treatment = engine.suggest_treatment_plan()
        report = engine.generate_report()
        
        return jsonify({
            'success': True,
            'diagnosis': disease,
            'confidence': round(confidence * 100, 2),
            'treatment': treatment,
            'report': report,
            'symptom_details': symptom_details  # Include in response
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    symptoms = [
        "Fever", "Cough", "Headache", "Fatigue", 
        "Nausea", "Shortness of Breath", "Sore Throat", 
        "Muscle Pain", "Loss of Taste", "Chest Pain",
        "Mucus", "Chills", "Runny Nose", "Body Aches",
        "Vomiting", "Diarrhea", "Rash", "Itching",
        "Dizziness", "Blurred Vision", "Wheezing",
        "Loss of Smell", "Sneezing", "Chest Tightness"
    ]
    return jsonify({'symptoms': symptoms})

if __name__ == '__main__':
    app.run(debug=True, port=5000)