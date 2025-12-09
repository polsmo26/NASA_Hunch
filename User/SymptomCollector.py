class SymptomCollector:
    """
    Handles user interaction and collects patient symptoms.
    """


    def __init__(self):
        self.patient_name = ""
        self.symptom_list = []
        self.duration_of_symptoms = {}
        self.severity_scores = {}


    def display_symptom_menu(self):
        """
        Displays a list of common symptoms for the user to select.
        """
        print("\n--- Symptom Menu ---")
        self.available_symptoms = [
            "Fever",
            "Cough",
            "Headache",
            "Fatigue",
            "Nausea",
            "Shortness of Breath",
            "Sore Throat",
            "Muscle Pain"
        ]
        for i, symptom in enumerate(self.available_symptoms, 1):
            print(f"{i}. {symptom}")
        print("0. Done selecting symptoms")


    def get_user_input(self):
        """
        Collects patient information, symptoms, duration, and severity.
        """
        self.patient_name = input("Enter patient name: ").strip()
        self.display_symptom_menu()


        while True:
            try:
                choice = int(input("\nSelect a symptom number (0 to finish): "))
                if choice == 0:
                    break
                elif 1 <= choice <= len(self.available_symptoms):
                    symptom = self.available_symptoms[choice - 1]
                    if symptom not in self.symptom_list:
                        self.symptom_list.append(symptom)
                        duration = input(f"Enter duration of {symptom} (e.g., '3 days'): ").strip()
                        self.duration_of_symptoms[symptom] = duration
                        severity = input(f"Enter severity of {symptom} (1-10): ").strip()
                        self.severity_scores[symptom] = int(severity)
                    else:
                        print("You already selected this symptom.")
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")


        print("\nSymptom collection completed.\n")


    def validate_symptoms(self):
        """
        Validates that symptoms and details have been collected correctly.
        Returns True if data is valid, False otherwise.
        """
        if not self.patient_name:
            print("Validation failed: Patient name is missing.")
            return False
        if not self.symptom_list:
            print("Validation failed: No symptoms selected.")
            return False


        for symptom in self.symptom_list:
            if symptom not in self.duration_of_symptoms or symptom not in self.severity_scores:
                print(f"Validation failed: Missing data for {symptom}.")
                return False


        print("All inputs validated successfully.")
        return True


if __name__ == "__main__":
    collector = SymptomCollector()
    collector.get_user_input()
    if collector.validate_symptoms():
        print("\nCollected Data:")
        print(f"Patient: {collector.patient_name}")
        for s in collector.symptom_list:
            print(f"- {s}: {collector.duration_of_symptoms[s]}, Severity {collector.severity_scores[s]}")
