import os
import sys
import csv


# Add project root to Python path
# ROOT = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(ROOT)
#ROOT = os.path.dirname(os.path.abspath(__file__))
#ROOT = os.path.dirname(ROOT)  # Go up one level to project root


# Add project root to Python path - FIXED VERSION
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level
sys.path.insert(0, project_root)  # Add project root to path


from Machine_Learning.MachineLearningEngine import MachineLearningEngine
from InferenceEngine import InferenceEngine
from KnowledgeBase import KnowledgeBase
from OutputModule import OutputModule
from SymptomCollector import SymptomCollector




class DiagnosisEngine:
    def __init__(self, probable_disease=None, confidence_level=0.0):
        """
        Initialize the DiagnosisEngine with optional probable disease and confidence level.
        """
        self.probable_disease = probable_disease
        self.confidence_level = confidence_level
        
        # FIXED: Use absolute paths to find the ML model
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "Machine_Learning", "ml_model.joblib")
        mlb_path = os.path.join(project_root, "Machine_Learning", "mlb.joblib")
        
        self.ml_engine = MachineLearningEngine(
            model_path=model_path,
            mlb_path=mlb_path
        )
        
        try:
            self.ml_engine.load()
            print("Machine Learning model loaded successfully!")
        except FileNotFoundError:
            print("ML model not found - run train_ml.py first")
            
            

    def generate_report(self):
        """
        Generates a diagnostic report summarizing the probable disease and confidence level.
        Returns a formatted string report.
        """
        if not self.probable_disease:
            return "No probable disease identified. Further analysis required."


        report = (
            f"Diagnosis Report:\n"
            f"- Probable Disease: {self.probable_disease}\n"
            f"- Confidence Level: {self.confidence_level * 100:.1f}%\n"
        )


        if self.confidence_level >= 0.8:
            report += "Diagnosis is highly reliable based on the current data."
        elif self.confidence_level >= 0.5:
            report += "Diagnosis is moderately reliable; additional tests are recommended."
        else:
            report += "Diagnosis is uncertain; further evaluation is strongly advised."


        return report


    def suggest_treatment_plan(self):
        """
        Suggests a treatment plan based on the identified probable disease.
        Returns a list of recommended actions or an advisory message.
        """
        if not self.probable_disease:
            return ["No treatment plan available without a confirmed diagnosis."]


        treatment_plans = {
            "Influenza": [
                "Rest and hydration",
                "Antiviral medication if prescribed",
                "Monitor for fever and respiratory distress"
            ],
            "Diabetes": [
                "Maintain a balanced diet",
                "Monitor blood glucose regularly",
                "Consult with an endocrinologist"
            ],
            "Hypertension": [
                "Reduce salt intake",
                "Exercise regularly",
                "Take prescribed antihypertensive medications"
            ],
            "COVID-19": [
                "Isolate and monitor symptoms",
                "Stay hydrated and rest",
                "Seek medical attention if symptoms worsen"
            ]
        }


        return treatment_plans.get(
            self.probable_disease,
            ["Consult a medical professional for a personalized treatment plan."]
        )


    def ml_predict(self, symptoms):
        """ Returns ML prediction list of (disease, probability)
        symptoms: list of user-entered symptoms """
        return self.ml_engine.predict(symptoms, top_k=3)


    def diagnose(self, symptom_list):
        """
        Try ML first (if model loaded), otherwise fallback to rule-based.
        Returns (disease, confidence).
        """
        if self.ml_engine.model is not None:
            try:
                ml_results = self.ml_engine.predict(symptom_list, top_k=1)
                predicted_disease, probability = ml_results[0]
                self.probable_disease = predicted_disease
                self.confidence_level = probability
                return predicted_disease, probability
            except Exception as e:
                print("ML prediction failed:", e)
                
        # Fallback to rule-based diagnosis
        kb = KnowledgeBase()
        ie = InferenceEngine(kb)
        ranked = ie.rank_possible_diagnoses(symptom_list)
        if ranked:
            disease, score = ranked[0]
            self.probable_disease = disease
            self.confidence_level = score
            return disease, score
        return None, 0.0


    def run(self):
        collector = SymptomCollector()
        collector.get_user_input()
        if not collector.validate_symptoms():
            print("Invalid input. Exiting...")
            return


        # Normalize symptoms to lowercase
        symptoms = [s.lower() for s in collector.symptom_list]


        # Rule-based inference
        kb = KnowledgeBase()
        inference_engine = InferenceEngine(kb)
        ranked_diagnoses = inference_engine.rank_possible_diagnoses(symptoms)
        rule_diagnosis = ranked_diagnoses[0][0] if ranked_diagnoses else None
        explanation = []
        if ranked_diagnoses:
            explanation.append(inference_engine.explain_reasoning(ranked_diagnoses[:1]))


        # Try ML prediction
        final_diagnosis = rule_diagnosis
        try:
            ml_results = self.ml_engine.predict(symptoms, top_k=1)
            ml_prediction, ml_confidence = ml_results[0]
            if ml_confidence > 0.6:
                final_diagnosis = ml_prediction
                explanation.append(f"ML model selected (confidence {ml_confidence:.2f}).")
            else:
                explanation.append("Rule-based engine selected (ML confidence too low).")
        except RuntimeError:
            explanation.append("Using rule-based diagnosis (ML model not available).")


        output = OutputModule(final_diagnosis, explanation)
        print(output.display_results())
        recommendations = output.print_recommendations()
        print("\n=== Recommendations ===")
        for rec in recommendations:
            print(f"- {rec}")




if __name__ == "__main__":
    engine = DiagnosisEngine()
    engine.run()
