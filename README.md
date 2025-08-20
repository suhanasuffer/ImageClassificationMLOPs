#Image Classification
This project implements a simple **image classification pipeline** using:
- **Scikit-learn (MLPClassifier)** for model training
- **NumPy & Pandas** for data handling
- **DVC** for reproducible pipelines and experiment tracking

#Project structure
image_classification/
├── data/ 
├── models/ 
├── reports/ 
├── src/ 
│ ├── data_processing.py
│ ├── model_training.py
│ ├── model_evaluation.py
├── params.yaml 
├── dvc.yaml 
├── dvc.lock 
├── requirements.txt 
└── README.md

#Run it
1. Clone the repository
```bash
git clone https://github.com/suhanasuffer/ImageClassificationMLOPs
cd image_classification

2. Install dependencies
pip install -r requirements.txt

3. Run the DVC pipeline
dvc repro

This will:

Process data
Train the model
Evaluate it
And finally, save metrics to reports/metrics.json.


