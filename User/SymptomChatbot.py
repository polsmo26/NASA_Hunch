class SymptomChatbot:
    def __init__(self):
        self.symptoms = []
        self.current_symptom = None

    def normalize(self, text):
        return text.lower().strip().replace(" ", "_")

    def ask(self, prompt):
        return input(f"Bot: {prompt}\nYou: ").strip()

    def start(self):
        print("Bot: Hi! I’ll ask a few questions about your symptoms.")
        while True:
            symptom = self.ask("What symptom are you experiencing? (or type 'done')")
            if symptom.lower() == "done":
                break

            symptom = self.normalize(symptom)
            entry = {"name": symptom}

            # Severity
            while True:
                sev = self.ask(f"On a scale of 1–10, how severe is your {symptom}?")
                if sev.isdigit() and 1 <= int(sev) <= 10:
                    entry["severity"] = int(sev)
                    break
                print("Bot: Please enter a number from 1 to 10.")

            # Duration
            while True:
                dur = self.ask(f"How many days have you had {symptom}?")
                if dur.isdigit() and int(dur) >= 0:
                    entry["duration_days"] = int(dur)
                    break
                print("Bot: Please enter a valid number of days.")

            self.symptoms.append(entry)

        return self.symptoms
    