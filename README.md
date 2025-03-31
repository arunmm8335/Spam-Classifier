# Spam Classifier

A machine learning-based spam classification application that uses a Flask web framework and a trained model to classify messages as spam or not.

## Table of Contents
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Files Overview](#files-overview)
- [Model Details](#model-details)
- [Evaluation](#evaluation)

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/arunmm8335/Spam-Classifier.git
cd Spam-Classifier
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment  
On **Windows**:
```bash
.env\Scriptsctivate
```
On **Mac/Linux**:
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```
This will install the necessary Python libraries like Flask, pandas, scikit-learn, and nltk.

---

## Setup

### 1. Download NLTK Stopwords  
When you run the app for the first time, the NLTK library will attempt to download the stopwords dataset. Ensure you're connected to the internet for this step.

### 2. Download the Spam Dataset  
Make sure the `spam.csv` file is placed inside the `data/` directory. The dataset can be downloaded from [Kaggle's Spam SMS Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

---

## Usage

### 1. Train the Model  
The model training process is handled by `spam_classifier.py`, which uses scikit-learn's machine learning algorithms. If you want to retrain or experiment with different algorithms, run:
```bash
python spam_classifier.py
```
This will:
- Load and preprocess the dataset.
- Train a spam classification model.
- Save the trained model as `spam_classifier.pkl` inside the `models/` directory.

### 2. Run the Flask App  
After training, run the Flask application to deploy the spam classifier as a web service:
```bash
python app/app.py
```
The app will start a local server, usually accessible at [http://127.0.0.1:5000](http://127.0.0.1:5000).

### 3. Use the Web Interface
- Navigate to the website.
- Enter a message in the text field and click **"Check"** to classify it as either **Spam** or **Ham** (Not Spam).

---

## Files Overview

```
Spam-Classifier/
│── app/                # Contains Flask web application files
│   ├── app.py          # Main Flask application
│   ├── templates/      # HTML templates (e.g., index.html)
│── data/               # Directory for storing the dataset (spam.csv)
│── models/             # Directory for saving the trained model (spam_classifier.pkl)
│── notebooks/          # Jupyter notebooks for model experimentation
│── spam_classifier.py  # Script for training the model
│── requirements.txt    # List of required Python libraries
```

---

## Model Details

The model uses **Logistic Regression** (or another chosen classifier) trained on features extracted from SMS text messages. Text features are created using **TF-IDF Vectorization**, a method of transforming text into numerical vectors that reflect word importance.

### Training Steps:
1. **Data Loading** – The dataset is loaded into a pandas DataFrame.
2. **Preprocessing** – The text data is cleaned (removing stopwords, punctuation, and converting to lowercase).
3. **Feature Extraction** – TF-IDF Vectorizer converts text into numerical features.
4. **Model Training** – A **Logistic Regression** model (or other classifier) is trained.
5. **Model Evaluation** – Performance is assessed using accuracy, precision, recall, and F1-score.

---

## Evaluation

The trained model is evaluated using key metrics:

- **Accuracy** – Overall classification accuracy.
- **Precision** – Correct positive predictions out of all positive predictions.
- **Recall** – Actual positive samples correctly identified.
- **F1-Score** – Harmonic mean of precision and recall.

### Example Evaluation Output:
```yaml
Accuracy: 0.9668
              precision    recall  f1-score   support
           0       0.96      1.00      0.98       965
           1       1.00      0.75      0.86       150

    accuracy                           0.97      1115
   macro avg       0.98      0.88      0.92      1115
weighted avg       0.97      0.97      0.96      1115
```

---

