from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


class OutputModule:
    def __init__(self, final_diagnosis=None, explanation_trace=None):
        """
        Initialize the OutputModule with a final diagnosis and an explanation trace.
        """
        self.final_diagnosis = final_diagnosis
        self.explanation_trace = explanation_trace or []


    def display_results(self):
        """
        Displays the final diagnosis and explanation trace to the user.
        Returns a formatted string with the complete diagnostic output.
        """
        if not self.final_diagnosis:
            return "No diagnosis available. Please ensure that the inference engine has processed the data."


        output = f"Final Diagnosis: {self.final_diagnosis}\n"
        output += "\nExplanation Trace:\n"


        if self.explanation_trace:
            for i, step in enumerate(self.explanation_trace, start=1):
                output += f"  {i}. {step}\n"
        else:
            output += "  No detailed explanation available.\n"


        return output


    def print_recommendations(self):
        """
        Prints or returns recommendations based on the diagnosis.
        """
        if not self.final_diagnosis:
            return ["No recommendations available. Please generate a diagnosis first."]


        recommendations = {
            "Influenza": [
                "Get plenty of rest and stay hydrated.",
                "Take prescribed antiviral medication if needed.",
                "Monitor for persistent fever or breathing issues."
            ],
            "Diabetes": [
                "Maintain a consistent blood sugar monitoring routine.",
                "Adopt a balanced diet and regular exercise plan.",
                "Consult your healthcare provider for medication adjustments."
            ],
            "Hypertension": [
                "Reduce sodium intake and avoid excessive stress.",
                "Engage in regular physical activity.",
                "Take prescribed blood pressure medication as directed."
            ],
            "COVID-19": [
                "Isolate to prevent transmission.",
                "Stay hydrated and rest adequately.",
                "Seek medical help if symptoms worsen."
            ]
        }


        return recommendations.get(
            self.final_diagnosis,
            ["Consult a healthcare provider for further evaluation and treatment."]
        )


    def export_to_pdf(self, filename="diagnosis_report.pdf"):
        """
        Exports the diagnosis report and explanation trace to a PDF file.
        """
        if not self.final_diagnosis:
            raise ValueError("Cannot export: No diagnosis available.")


        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter


        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 72, "Diagnosis Report")


        c.setFont("Helvetica", 12)
        c.drawString(72, height - 100, f"Final Diagnosis: {self.final_diagnosis}")


        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, height - 130, "Explanation Trace:")


        y = height - 150
        for step in self.explanation_trace:
            c.setFont("Helvetica", 11)
            c.drawString(90, y, f"- {step}")
            y -= 15
            if y < 72:  # new page if text goes beyond bottom
                c.showPage()
                y = height - 72


        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y - 20, "Recommendations:")
        y -= 40


        for rec in self.print_recommendations():
            c.setFont("Helvetica", 11)
            c.drawString(90, y, f"- {rec}")
            y -= 15
            if y < 72:
                c.showPage()
                y = height - 72


        c.save()
        return f"Diagnosis report exported as '{filename}'"
    def display_ml_predictions(self, ml_results):
        print("\n=== Machine Learning Predictions ===")
        for disease, prob in ml_results:
            print(f"{disease}: {prob:.2f}")
