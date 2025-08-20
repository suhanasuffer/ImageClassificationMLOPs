import joblib
import numpy as np
import yaml
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    #loading test data
    X_test = np.load("data/features/X_test_selected.npy")
    y_test = np.load("data/features/y_test.npy")

    #laoding trained model
    model = joblib.load("models/model.pkl")

    #prediction
    y_pred = model.predict(X_test)

    #metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)


    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)

    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation results saved to reports/metrics.json")

if __name__ == "__main__":
    evaluate_model()
