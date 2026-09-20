# Loan Prediction Data Analysis System

**Data Analysis Essentials — Review 1 and Review 2**

A Streamlit application that trains and uses the Review-2 Logistic Regression model on the real `LoanApproval.csv` dataset.

## Model pipeline

- Renames `cibil_score` to `credit_score`
- Removes duplicate rows and excludes `loan_id`
- Encodes Graduate/Not Graduate, Yes/No, and Approved/Rejected exactly as in the notebook
- Uses the specified 11 features in their original order
- Uses a stratified 80:20 split with `random_state=42`
- Trains `LogisticRegression(max_iter=1000, random_state=42)` without scaling
- Loads `model/loan_model.pkl` when available and retrains from the CSV if it is missing or incompatible

## Run on Windows in VS Code

Open the project folder in VS Code, then run these commands in its PowerShell terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, the app can be started directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The browser normally opens at `http://localhost:8501`.

## Project files

- `app.py` — Streamlit user interface
- `loan_logic.py` — dataset preparation, model training, persistence, and prediction helpers
- `LoanApproval.csv` — original project dataset
- `model/loan_model.pkl` — trained model bundle generated from the dataset
- `requirements.txt` — required Python packages
- `.streamlit/config.toml` — application theme
- `test_app.py` — automated model and Streamlit smoke tests

The original notebook is not modified by this application.

## Streamlit Community Cloud deployment

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Create an app from `rohitbandi0311-afk/loan-prediction-system`.
3. Select branch `main` and main file path `app.py`.
4. Deploy. This application does not require secrets.

## Review-1 project background

An academic project focused on understanding loan-application data through data collection, preprocessing, exploratory data analysis (EDA), visualization, and identification of patterns related to loan approval.

## Objectives

- Understand the loan dataset and its features.
- Check data quality, missing values, duplicates, and data types.
- Perform exploratory data analysis.
- Study relationships between applicant and financial features.
- Create visualizations to identify useful patterns.
- Prepare the dataset for the next phase: loan prediction modeling.

## Dataset

The dataset contains **4,269 loan application records and 13 columns**, including applicant details, income, loan information, CIBIL score, asset values, and loan status.

The raw dataset is stored in:

`data/LoanApproval.csv`

## Analysis Performed

- Dataset structure and data-type inspection
- Missing-value and duplicate checks
- Descriptive statistics
- Loan-status distribution
- Credit-group analysis
- Self-employment analysis
- Income and loan-amount analysis
- Correlation analysis and heatmap
- Data visualizations using Matplotlib and Seaborn

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Google Colab / Jupyter Notebook

## Repository Structure

```text
loan-prediction-system/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── LoanApproval.csv
│   └── README.md
│
├── notebooks/
│   └── README.md
│
├── docs/
│   └── project_summary.md
│
└── presentation/
    ├── Loan_Prediction_System_Review1.pptx
    └── README.md
```

## Project Workflow

```text
Dataset Collection
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis (EDA)
        ↓
Visualization & Analysis
        ↓
Insights
        ↓
Loan Prediction Model — Next Phase
```

## Review-2 phase

Review 2 adds categorical encoding, feature selection, Logistic Regression model development, model evaluation, and the interactive Streamlit prediction application documented above.

## Team

- **B. Rohith** — 25B11DS045
- **Ali Raza Hasan** — 25B11DS016
- **K. Anand Sai** — 25B11DS286
- **Vishnu Vardhan** — 25B11DS212

## Academic Project

**University:** Aditya University  
**Course:** Data Analysis Essentials  
**Project:** Loan Prediction System  
**Review:** Review 2 — Machine Learning Application

## License

This project is intended for academic and educational use.
