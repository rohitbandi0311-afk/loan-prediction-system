# Loan Prediction Data Analysis System

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
