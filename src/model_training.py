import yaml
import joblib
import numpy as np
import os
from sklearn.neural_network import MLPClassifier

def train_model():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    training_params = params["training"]

    #loading data after feature selection
    X_train = np.load("data/features/X_train_selected.npy")
    y_train = np.load("data/features/y_train.npy")

    #defining model
    model = MLPClassifier(
        hidden_layer_sizes=tuple(training_params["hidden_layer_sizes"]),
        activation=training_params["activation"],
        solver=training_params["solver"],
        alpha=training_params["alpha"],
        learning_rate_init=training_params["learning_rate_init"],
        max_iter=training_params["max_iter"],
        random_state=training_params["random_state"],
    )

    #training model
    model.fit(X_train, y_train)

    #saving trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("Model training complete. Saved to models/model.pkl")

if __name__ == "__main__":
    train_model()
