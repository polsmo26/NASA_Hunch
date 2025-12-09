import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


class MachineLearningEngine:
    """
    Simple ML engine that trains a RandomForest on symptom sets (multi-label
    one-hot encoded) to predict a single disease label.
    """


    def __init__(self, model_path="ml_model.joblib", mlb_path="mlb.joblib"):
        self.model_path = model_path
        self.mlb_path = mlb_path
        self.model = None
        self.mlb = None


    # ---------- Data helpers ----------
    @staticmethod
    def _parse_symptom_string(symptom_str):
        """
        Convert 'fever;cough' -> ['fever','cough'] with normalization.
        """
        if pd.isna(symptom_str):
            return []
        parts = [s.strip() for s in str(symptom_str).split(";") if s.strip()]
        return [MachineLearningEngine.normalize_symptom(s) for s in parts]


    @staticmethod
    def normalize_symptom(s):
        """
        Normalize a symptom string to match your KB format:
        lowercased and spaces -> underscore, remove extra chars.
        """
        return s.lower().strip().replace(" ", "_")


    # ---------- Training ----------
    def train_from_csv(self, csv_path, test_size=0.5, random_state=42, verbose=True):
        """
        Train a RandomForest classifier from a CSV with columns: 'symptoms','disease'.
        Symptoms should be semicolon-separated.
        Saves model and MultiLabelBinarizer to files.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found")
        df = pd.read_csv(csv_path)
        if 'symptoms' not in df.columns or 'disease' not in df.columns:
            raise ValueError("CSV must contain 'symptoms' and 'disease' columns")
        # Parse symptom lists
        df['symptom_list'] = df['symptoms'].apply(self._parse_symptom_string)
        # Create MultiLabelBinarizer (one-hot encoding of symptoms)
        self.mlb = MultiLabelBinarizer(sparse_output=False)
        X = self.mlb.fit_transform(df['symptom_list'])
        y = df['disease'].astype(str)
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        # Model — simple RandomForest
        self.model = RandomForestClassifier(n_estimators=200, random_state=random_state)
        self.model.fit(X_train, y_train)
        # Evaluation (simple)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        if verbose:
            print(f"Training complete — test accuracy: {acc:.3f}")
            print("Classification report:")
            print(classification_report(y_test, preds))


        # Persist
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.mlb, self.mlb_path)
        if verbose:
            print(f"Saved model -> {self.model_path}")
            print(f"Saved symptom binarizer -> {self.mlb_path}")


        return acc


    # ---------- Load ----------
    def load(self):
        """
        Load model and mlb from disk. Raises if not found.
        """
        if not os.path.exists(self.model_path) or not os.path.exists(self.mlb_path):
            raise FileNotFoundError("Model or mlb file not found. Train first or provide files.")
        self.model = joblib.load(self.model_path)
        self.mlb = joblib.load(self.mlb_path)


    # ---------- Predict ----------
    def predict(self, symptom_list, top_k=1):
        """
        Predict disease from a list of symptoms.
        Returns a list of (disease, probability) sorted by probability desc.
        symptom_list may be Title Case or lower—normalize inside.
        """
        if self.model is None or self.mlb is None:
            raise RuntimeError("Model not loaded. Call load() or train_from_csv().")
        normalized = [self.normalize_symptom(s) for s in symptom_list]
        X = self.mlb.transform([normalized])  # shape (1, n_features)
        probs = self.model.predict_proba(X)[0]  # shape (n_classes,)
        classes = self.model.classes_
        class_probs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        # return top_k
        return class_probs[:top_k]
