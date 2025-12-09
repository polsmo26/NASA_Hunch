import os
from MachineLearningEngine import MachineLearningEngine


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "training_data.csv")

    ml = MachineLearningEngine(
        model_path=os.path.join(base, "ml_model.joblib"),
        mlb_path=os.path.join(base, "mlb.joblib"),
    )

    acc = ml.train_from_csv(csv_path, test_size=0.5)
    print("Done. Test accuracy:", acc)
