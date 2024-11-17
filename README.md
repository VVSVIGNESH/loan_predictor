# Loan Prediction Project

This project builds a loan prediction model using various machine learning algorithms. It predicts loan eligibility based on applicant attributes like credit score, income, marital status, etc. The project follows a complete workflow from data preprocessing to model evaluation, showcasing the implementation of different classification models and their performance.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project aims to predict whether a loan will be approved based on various applicant attributes. The model takes into account factors such as income, credit history, marital status, and more. It uses multiple machine learning algorithms and evaluates their performance on a given dataset.

## Dataset
The dataset used in this project contains the following attributes:

| Attribute          | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| **Gender**         | Male/Female                                                                  |
| **Married**        | Applicant's marital status (Yes/No)                                          |
| **Dependents**     | Number of dependents                                                          |
| **Education**      | Graduate/Not Graduate                                                         |
| **Self_Employed**  | Yes/No                                                                        |
| **ApplicantIncome**| Monthly income of the applicant                                               |
| **CoapplicantIncome**| Monthly income of the co-applicant                                          |
| **LoanAmount**     | Loan amount in thousands                                                      |
| **Loan_Amount_Term**| Term of the loan in months                                                   |
| **Credit_History** | 1: Credit history meets guidelines, 0: Does not meet guidelines              |
| **Property_Area**  | Urban/Semi-Urban/Rural                                                       |
| **Loan_Status**    | Y: Loan approved, N: Loan not approved (Target variable)                      |

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/loan-prediction-project.git
    cd loan-prediction-project
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Place the dataset (`loan-predictionUC.csv` or `loan-predictionUC.xlsx`) in the project directory.

## Usage
To run the project, use either Jupyter Notebook or the Python script:

- **For Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

- **For Python script**:
    ```bash
    python loan_prediction.py
    ```

### Workflow
The code will:
1. Load and preprocess the dataset.
2. Handle missing values and encode categorical variables.
3. Perform feature engineering to enhance prediction accuracy.
4. Train and evaluate multiple machine learning models.
5. Display performance metrics and visualizations for each model.

## Modeling
The following models are evaluated in this project:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Support Vector Machine (SVM)**
- **Linear Regression**

Each model is tested for accuracy, and performance metrics such as confusion matrices and classification reports are generated. Additionally, visualization techniques are employed to display decision boundaries, feature importances, and model-specific plots.

## Results
- **Best Model**: The **Logistic Regression** model performed the best, achieving the highest accuracy score and proving to be the most suitable model for predicting loan eligibility with this dataset.
- **Key Metrics**: The final model's accuracy, confusion matrix, and classification report are provided in the code.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Make your changes.
3. Submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
