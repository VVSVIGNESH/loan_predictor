# loan_predictor
# Built a loan prediction model using Python and Machine Learning with 79% accuracy, utilizing features like credit score and income..
Loan Prediction Project:
This repository contains a machine learning project that predicts loan eligibility based on various applicant attributes. The project follows a complete workflow from data preprocessing to model evaluation using several classification algorithms.

Table of Contents:
Overview
Dataset
Installation
Usage
Modeling
Results
Contributing
License

Overview:
This project involves building and evaluating various machine learning models to predict loan approval based on applicant details such as income, credit history, and employment status. The dataset used is processed to handle missing values, encode categorical variables, and engineer new features to improve prediction accuracy.

Dataset:
The dataset used in this project contains the following attributes:

Gender: Male/Female
Married: Applicant's marital status
Dependents: Number of dependents
Education: Graduate/Not Graduate
Self_Employed: Yes/No
ApplicantIncome: Income of the applicant
CoapplicantIncome: Income of the co-applicant
LoanAmount: Loan amount in thousands
Loan_Amount_Term: Term of the loan in months
Credit_History: Credit history meets guidelines (1) or not (0)
Property_Area: Urban/Semi-Urban/Rural
Loan_Status: Loan approved (Y) or not (N) - (Target variable)

Installation:
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/loan-prediction-project.git
cd loan-prediction-project
Create a virtual environment and activate it:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Place the dataset (loan-predictionUC.csv (3) (2).xlsx) in the project directory.

Usage:
Run the Jupyter notebook or Python script to execute the full workflow:

For Jupyter notebook: jupyter notebook
For Python script: python loan_prediction.py
The code will:

Load and preprocess the dataset.
Handle missing values and encode categorical variables.
Perform feature engineering.
Train and evaluate multiple machine learning models.
Display performance metrics and visualizations for each model.
Modeling
The following models are evaluated in this project:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Gradient Boosting Classifier
Support Vector Machine (SVM)
Linear Regression
Each model is tested for accuracy, and performance metrics such as confusion matrices and classification reports are generated. Visualization techniques are also employed to display decision boundaries, feature importances, and model-specific plots.

Results:
The Logistic Regression model performed the best with the highest accuracy score, making it the most suitable for predicting loan eligibility in this dataset. The final model and its performance metrics are discussed in detail in the code.

Contributing:
Contributions are welcome! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.

License:
This project is licensed under the MIT License. See the LICENSE file for more details.
