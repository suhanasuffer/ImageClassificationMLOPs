import numpy as np
import os
import yaml
from sklearn.feature_selection import VarianceThreshold

#loading params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

feature_params = params["features"]

def select_features():
    #loading preprocessed data
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    #feature selection using variance threshold
    selector = VarianceThreshold(threshold=feature_params["variance_threshold"])
    X_train_selected = selector.fit_transform(X_train)
    X_test_selected = selector.transform(X_test)

    #creating directory if not exists
    os.makedirs("data/features", exist_ok=True)

    #saving selected features
    np.save("data/features/X_train_selected.npy", X_train_selected)
    np.save("data/features/X_test_selected.npy", X_test_selected)
    np.save("data/features/y_train.npy", y_train)
    np.save("data/features/y_test.npy", y_test)

    print(f"Feature selection done. New shape: {X_train_selected.shape}")

if __name__ == "__main__":
    select_features()
