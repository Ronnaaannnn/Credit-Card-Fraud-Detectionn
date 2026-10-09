# Credit Card Fraud Detection: Flask Web App

An end-to-end machine learning project that detects fraudulent credit card transactions using a Random Forest classifier and serves predictions through a Flask web application. The application includes a training dashboard, model evaluation graphs, and an interactive transaction prediction form.

**Dataset:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Problem Statement

The dataset contains credit card transactions made by European cardholders over two days in September 2013. It contains **284,807 transactions, of which only 492 are fraudulent (0.172%)**.

Due to this extreme class imbalance, accuracy alone is not a reliable evaluation metric. A model that classifies every transaction as legitimate would achieve approximately 99.83% accuracy while failing to detect any fraud.

Therefore, this project emphasizes **Area Under the Precision-Recall Curve (AUPRC)**, alongside other metrics that evaluate fraud detection performance.

### Dataset Features

* `Time`: Seconds elapsed since the first transaction, ranging from 0 to 172,792.
* `V1`–`V28`: Anonymized features obtained through Principal Component Analysis (PCA).
* `Amount`: Transaction amount.
* `Class`: Target variable, where `0` represents a legitimate transaction and `1` represents a fraudulent transaction.

## Methodology

1. **Data Loading:** Load the credit card transaction dataset and validate the required features.
2. **Data Splitting:** Split the dataset into 70% training data and 30% testing data using stratified sampling to preserve the fraud-to-legitimate transaction ratio.
3. **Model Training:** Train a Random Forest classifier using `class_weight="balanced_subsample"` to help address class imbalance.
4. **Model Evaluation:** Evaluate performance on the held-out test set using AUPRC, ROC-AUC, precision, recall, and F1-score.
5. **Artifact Generation:** Save the trained model, evaluation metrics, test samples, and visualization plots.
6. **Web Application:** Use Flask to provide an interactive dashboard and a transaction prediction interface.

## Evaluation and Visualizations

The application generates the following visualizations:

* Class distribution on a logarithmic scale.
* Precision-Recall curve.
* Receiver Operating Characteristic (ROC) curve.
* Confusion matrix.
* Top 15 features ranked by importance.
* Fraudulent transactions over time.
* Transaction amount distribution for legitimate and fraudulent transactions.

These visualizations help evaluate the model's performance, understand its predictions, and examine patterns in the dataset.

## Web Application Features

### Dashboard (`/`)

* Displays key model evaluation metrics.
* Shows training and testing dataset information.
* Presents all generated evaluation graphs.
* Displays the classification report.
* Provides a button to train or retrain the model.

### Transaction Prediction (`/predict`)

* Enter all 30 transaction features manually.
* Paste a complete transaction row.
* Load a legitimate or fraudulent example from the test dataset.
* View the predicted fraud probability and classification verdict.

### Input Validation

* `Time` must be between 0 and 172,792 seconds.
* `Amount` must be greater than or equal to zero.
* All transaction feature values must be numeric.

**Note:** The prediction probability is a model output and should not automatically be interpreted as a calibrated real-world fraud risk.

## Project Structure

```text
fraud_app/
├── app.py                  # Flask routes: dashboard, training, prediction
├── model.py                # Model training, evaluation, and graph generation
├── requirements.txt        # Python dependencies
├── templates/
│   ├── base.html            # Shared page layout
│   ├── index.html           # Dashboard
│   └── predict.html         # Transaction prediction interface
├── static/
│   └── plots/               # Generated evaluation graphs
└── artifacts/
    ├── model.joblib         # Trained machine learning model
    ├── metrics.json         # Model evaluation metrics
    └── test_samples.csv     # Sample transactions for testing
```

## Installation and Usage

### Prerequisites

* Python 3.10 or later.
* pip package manager.
* The credit card fraud detection dataset from Kaggle.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd fraud_app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download `creditcard.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project directory.

### 4. Train the Model

```bash
python model.py
```

This step trains the model, evaluates its performance, and generates the required artifacts and visualizations.

### 5. Start the Flask Application

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

## Model Performance

Run the training script to generate the actual evaluation results. Replace the placeholders below with the metrics produced by your model.

| Metric          | Score          |
| --------------- | -------------- |
| AUPRC           | To be measured |
| ROC-AUC         | To be measured |
| Fraud Precision | To be measured |
| Fraud Recall    | To be measured |
| Fraud F1-Score  | To be measured |

**Important:** These values must be filled in using actual test-set results. No performance scores are assumed or fabricated.

## Future Improvements

* Compare Random Forest with XGBoost and LightGBM.
* Experiment with SMOTE, random undersampling, and other imbalance-handling techniques.
* Tune the classification threshold using a cost-sensitive evaluation approach.
* Investigate transaction amount and potential financial losses when defining fraud detection costs.
* Add model explainability using SHAP.
* Improve probability calibration and threshold selection.
* Add automated tests and stronger input validation.
* Containerize the application using Docker.
* Deploy the application using a platform such as Render, Railway, or AWS.

## Technology Stack

* **Programming Language:** Python
* **Web Framework:** Flask
* **Machine Learning:** scikit-learn
* **Data Processing:** pandas, NumPy
* **Data Visualization:** Matplotlib
* **Model Persistence:** joblib
* **Frontend Templating:** Jinja2
* **Dataset:** Kaggle Credit Card Fraud Detection

## License and Credits

The dataset was provided by the Machine Learning Group at Université Libre de Bruxelles (ULB) and Worldline.

The dataset is distributed under the Database Contents License (DbCL). Refer to the original [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) for the dataset's licensing details and terms of use.

This project is intended for educational and research purposes. Its predictions should not be treated as a substitute for a production-grade fraud detection system.
