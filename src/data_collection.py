import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def read_params(config_path="params.yaml"):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def collect_data(config_path="params.yaml"):
    config = read_params(config_path)
    data_params = config["data"]
    processing_params = config["processing"]

    #loading dataset
    digits = load_digits()
    images, labels = digits.images, digits.target

    #output directory
    raw_dir = os.path.join("data", "raw")
    os.makedirs(f"{raw_dir}/images", exist_ok=True)

    #saving images
    for idx, img in enumerate(images):
        plt.imsave(f"{raw_dir}/images/{idx}.png", img, cmap="gray")

    #saving labels csv
    df = pd.DataFrame({
        "filename": [f"{i}.png" for i in range(len(images))],
        "label": labels
    })
    df.to_csv(f"{raw_dir}/labels.csv", index=False)

    print(f"Saved {len(images)} images and labels to {raw_dir}")


if __name__ == "__main__":
    collect_data()
