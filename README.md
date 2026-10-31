# 🍷 Wine Quality Prediction Using Machine Learning

## 🚀 Live Demo

Try the deployed Wine Quality Prediction application:

**[🍷 Launch Wine Quality Prediction App](https://wine-quality-ml-haha93kritn8rqs5kdv639.streamlit.app/)**

The application provides two independent Machine Learning predictions:

- **Classification:** Predicts whether wine quality is below 6 or at least 6.
- **Regression:** Predicts a continuous wine quality score.

Built using Python, Scikit-learn and Streamlit.

## Project Overview

This project uses supervised machine learning techniques to predict wine quality based on its physicochemical properties.

It implements two machine learning approaches:

- **Classification:** Predicts whether wine quality is below 6 or 6 and above.
- **Regression:** Predicts the numerical wine quality score.

The project includes data preprocessing, exploratory data analysis, model benchmarking, cross-validation, hyperparameter tuning, model evaluation, and an interactive Streamlit application.

## Objectives

- Analyze wine characteristics using exploratory data analysis.
- Build classification and regression models.
- Compare multiple machine learning algorithms.
- Optimize model performance using RandomizedSearchCV.
- Evaluate models using appropriate performance metrics.
- Develop an interactive wine quality prediction application.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Application | Streamlit |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## Dataset

The project uses wine quality data containing physicochemical characteristics of red and white wine.

**Input features:**
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Wine type

**Target variable:** Quality

## Machine Learning Approaches

### 1. Classification

The target variable is converted into a binary classification problem:

- 0: Wine quality below 6
- 1: Wine quality 6 or above

**Evaluation metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### 2. Regression

The original wine quality score is used as a continuous numerical target.

**Evaluation metrics:**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Model Performance

### Classification Results

| Metric | Baseline | Tuned |
|---|---:|---:|
| Accuracy | 83.3% | 84.2% |
| Precision | 84.6% | 85.3% |
| Recall | 90.0% | 90.5% |
| F1-score | 87.2% | 87.9% |

### Regression Results

| Metric | Baseline | Tuned |
|---|---:|---:|
| RMSE | 0.615 | 0.595 |
| MAE | 0.448 | 0.428 |
| R² Score | 0.488 | 0.521 |

**Key observations:**
- Classification accuracy improved from 83.3% to 84.2%.
- Classification F1-score improved from 87.2% to 87.9%.
- Regression RMSE decreased from 0.615 to 0.595.
- Regression R² increased from 0.488 to 0.521.

## Hyperparameter Optimization

RandomizedSearchCV was used to explore hyperparameter combinations for the selected classification and regression models.

Cross-validation was used during model tuning to evaluate candidate configurations.

## Project Structure

```text
wine-quality-ml/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── raw/
│
├── models/
│   ├── wine_quality_classifier_tuned.joblib
│   └── wine_quality_regressor_tuned.joblib
│
├── reports/
│
└── Wine_Quality_ML_Project.ipynb
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd wine-quality-ml
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**macOS/Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

## Application Features

- Select red or white wine.
- Enter physicochemical characteristics.
- Predict wine quality using classification.
- Estimate numerical wine quality using regression.
- View classification probability when supported by the model.

## Future Improvements

- Experiment with additional machine learning algorithms.
- Improve generalization through further model validation.
- Add model explainability and feature importance analysis.
- Deploy the application to a cloud hosting platform.
- Build an API using FastAPI.

## Author

**K. R. Saravanadeep**

- GitHub: https://github.com/saravanadeep
- LinkedIn: https://www.linkedin.com/in/k-r-saravanadeep-40249b306/

---

*This project was developed for practical learning and demonstration of supervised machine learning techniques.*
