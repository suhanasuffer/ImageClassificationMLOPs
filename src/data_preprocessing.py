import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib.image import imread


def read_params(config_path="params.yaml"):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def process_data(config_path="params.yaml"):
    config = read_params(config_path)
    data_params = config["data"]
    processing_params = config["processing"]

    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    #loading labels
    labels_df = pd.read_csv(os.path.join(raw_dir, "labels.csv"))

    X, y = [], []
    for _, row in labels_df.iterrows():
        img_path = os.path.join(raw_dir, "images", row["filename"])
        img = imread(img_path)

        #normalizing
        img = img / 16.0

        #flattening
        X.append(img.flatten())
        y.append(row["label"])

    X = np.array(X)
    y = np.array(y)

    #splitting data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,test_size=processing_params["test_size"],
        random_state=processing_params["random_state"]
    )

    #saving processed datasets
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)

    print(f"Processed data saved to {processed_dir}")


if __name__ == "__main__":
    process_data()
