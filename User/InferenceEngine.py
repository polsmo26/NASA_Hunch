from KnowledgeBase import KnowledgeBase


class InferenceEngine:
    """
    Processes user symptoms against the KnowledgeBase to infer
    possible diagnoses with confidence scores and explanations.
    """


    def __init__(self, knowledge_base: KnowledgeBase):
        self.ruleset = knowledge_base.rules
        self.disease_data = knowledge_base.diseases
        self.matched_rules = []


    def match_symptoms_to_rules(self, symptom_list):
        """
        Matches the provided symptoms to rules in the knowledge base.
        Stores all matched rules in self.matched_rules.
        """
        self.matched_rules = []
        for rule in self.ruleset:
            match_count = len(set(rule["IF"]) & set(symptom_list))
            if match_count > 0:
                confidence = match_count / len(rule["IF"])
                self.matched_rules.append({
                    "rule": rule,
                    "match_count": match_count,
                    "confidence": round(confidence, 2)
                })
        return self.matched_rules


    def calculate_confidence_score(self, disease_name, symptom_list):
        """
        Calculates a confidence score for how likely a disease is
        given the user’s symptoms and rule matches.
        """
        if disease_name not in self.disease_data:
            return 0.0


        known_symptoms = self.disease_data[disease_name]
        match_count = len(set(symptom_list) & set(known_symptoms))
        confidence = match_count / len(known_symptoms)
        return round(confidence, 2)


    def rank_possible_diagnoses(self, symptom_list):
        """
        Ranks diseases by confidence score, based on symptom overlap.
        Returns a sorted list of tuples: (disease_name, confidence_score)
        """
        ranked = []
        for disease in self.disease_data:
            score = self.calculate_confidence_score(disease, symptom_list)
            if score > 0:
                ranked.append((disease, score))


        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


    def explain_reasoning(self, top_diagnoses):
        """
        Provides human-readable explanations of how diagnoses were made,
        inspired by MYCIN’s ‘WHY?’ feature.
        """
        explanations = []
        for disease, confidence in top_diagnoses:
            explanation = (
                f"Diagnosis: {disease}\n"
                f"→ Confidence: {confidence * 100:.0f}%\n"
                f"→ Based on matching known symptoms: "
                f"{', '.join(self.disease_data[disease])}\n"
            )
            explanations.append(explanation)
        return "\n".join(explanations)




# Example usage
if __name__ == "__main__":
    kb = KnowledgeBase()
    engine = InferenceEngine(kb)


    user_symptoms = ["fever", "fatigue", "cough"]
    print("User symptoms:", user_symptoms)


    engine.match_symptoms_to_rules(user_symptoms)
    ranked = engine.rank_possible_diagnoses(user_symptoms)


    print("\nRanked Possible Diagnoses:")
    for d, c in ranked:
        print(f"- {d}: {c * 100:.0f}% confidence")


    print("\nExplanation:")
    print(engine.explain_reasoning(ranked[:2]))
