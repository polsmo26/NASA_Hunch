class KnowledgeBase:
    """
    Stores medical knowledge, including diseases, associated symptoms,
    and rules for basic diagnostic reasoning.
    """


    def __init__(self):
        # Initialize diseases and rule set
        self.diseases = {
            "Influenza": ["fever", "cough", "sore_throat", "fatigue"],
            "COVID-19": ["fever", "loss_of_taste", "fatigue", "cough"],
            "Bronchitis": ["cough", "chest_pain", "mucus", "fatigue"]
        }


        self.rules = [
            {"IF": ["fever", "cough", "sore_throat"], "THEN": "Influenza"},
            {"IF": ["fever", "loss_of_taste", "fatigue"], "THEN": "COVID-19"},
            {"IF": ["cough", "chest_pain", "mucus"], "THEN": "Bronchitis"}
        ]


    def load_rules(self, new_rules):
        """
        Loads new diagnostic rules into the knowledge base.
        """
        if isinstance(new_rules, list):
            self.rules.extend(new_rules)
            print(f"{len(new_rules)} new rules added successfully.")
        else:
            print("Error: Rules must be provided as a list of dictionaries.")


    def update_rules(self, disease_name, new_symptoms):
        """
        Updates existing disease information or adds a new disease entry.
        """
        if disease_name in self.diseases:
            self.diseases[disease_name].extend(
                [s for s in new_symptoms if s not in self.diseases[disease_name]]
            )
            print(f"Updated symptoms for {disease_name}.")
        else:
            self.diseases[disease_name] = new_symptoms
            print(f"Added new disease: {disease_name}.")


    def retrieve_possible_diseases(self, symptoms):
        """
        Given a list of symptoms, returns possible matching diseases.
        """
        possible_diseases = []


        # Compare symptoms to rules
        for rule in self.rules:
            if all(symptom in symptoms for symptom in rule["IF"]):
                possible_diseases.append(rule["THEN"])


        # If no rule matches exactly, fall back on partial matches
        if not possible_diseases:
            for disease, known_symptoms in self.diseases.items():
                match_count = len(set(symptoms) & set(known_symptoms))
                if match_count >= len(known_symptoms) // 2:
                    possible_diseases.append(disease)


        return possible_diseases




if __name__ == "__main__":
    kb = KnowledgeBase()


    print("Known Diseases:")
    print(list(kb.diseases.keys()))


    user_symptoms = ["fever", "fatigue", "loss_of_taste"]
    print("\nUser Symptoms:", user_symptoms)
    possible = kb.retrieve_possible_diseases(user_symptoms)


    print("\nPossible Diagnoses:")
    for disease in possible:
        print("-", disease)
