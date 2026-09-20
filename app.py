"""Streamlit application for the Loan Prediction Data Analysis System."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from loan_logic import (
    FEATURES,
    LoanDataError,
    load_model_bundle,
    predict_applicant,
    save_model_bundle,
    train_model,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "LoanApproval.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "model" / "loan_model.pkl"
MODEL_CANDIDATES = (
    DEFAULT_MODEL_PATH,
    BASE_DIR / "loan_model.pkl",
    BASE_DIR / "model.pkl",
    BASE_DIR / "loan_prediction_model.pkl",
)
DEFAULT_INPUTS = {
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 5_000_000,
    "loan_amount": 15_000_000,
    "loan_term": 10,
    "credit_score": 650,
    "residential_assets_value": 5_000_000,
    "commercial_assets_value": 3_000_000,
    "luxury_assets_value": 10_000_000,
    "bank_asset_value": 4_000_000,
}

st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

for input_key, default_value in DEFAULT_INPUTS.items():
    st.session_state.setdefault(input_key, default_value)


@st.cache_resource(show_spinner="Preparing the Logistic Regression model…")
def get_model_bundle() -> dict:
    """Load a compatible saved model, or train and cache one from the CSV."""
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            try:
                return load_model_bundle(candidate)
            except (OSError, ValueError, TypeError, LoanDataError):
                continue

    bundle = train_model(DATA_PATH)
    try:
        save_model_bundle(bundle, DEFAULT_MODEL_PATH)
    except OSError:
        # A read-only deployment can still use the in-memory trained model.
        pass
    return bundle


def rupees(value: int) -> str:
    return f"₹{value:,.0f}"


def validate(values: dict[str, int]) -> list[str]:
    errors: list[str] = []
    if values["no_of_dependents"] < 0:
        errors.append("Number of dependents cannot be negative.")
    if values["income_annum"] < 0:
        errors.append("Annual income cannot be negative.")
    if values["loan_amount"] < 0:
        errors.append("Loan amount cannot be negative.")
    if values["loan_term"] <= 0:
        errors.append("Loan term must be greater than zero.")
    if not 300 <= values["credit_score"] <= 900:
        errors.append("Credit score must be between 300 and 900.")
    for field in (
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ):
        if values[field] < 0:
            errors.append("Asset values cannot be negative.")
            break
    return errors


def reset_form() -> None:
    """Restore every applicant widget to its initial value."""
    for key, value in DEFAULT_INPUTS.items():
        st.session_state[key] = value


st.title("🏦 Loan Prediction System")
st.markdown("### AI-Based Loan Approval Prediction using Machine Learning")
st.caption(
    "Enter the applicant's financial and personal details to receive a prediction "
    "from the trained Logistic Regression model."
)

try:
    bundle = get_model_bundle()
except FileNotFoundError:
    st.error(
        "LoanApproval.csv is missing. Place it in the same folder as app.py and "
        "restart the application.",
        icon=":material/error:",
    )
    st.stop()
except (OSError, ValueError, TypeError, LoanDataError):
    st.error(
        "The model could not be prepared from the dataset. Please verify that "
        "LoanApproval.csv has the original Review-2 columns and values.",
        icon=":material/error:",
    )
    st.stop()

accuracy = float(bundle.get("metrics", {}).get("accuracy", 0.8033))
precision = float(bundle.get("metrics", {}).get("precision", 0.7942))
recall = float(bundle.get("metrics", {}).get("recall", 0.9228))
f1_score = float(bundle.get("metrics", {}).get("f1", 0.8537))
record_count = int(bundle.get("record_count", 4269))

with st.sidebar:
    st.header("Project overview", icon=":material/analytics:")
    st.markdown("**Loan Prediction Data Analysis System**")
    st.caption("College Review-2 project")
    with st.container(border=True):
        st.markdown("**Machine Learning Model**")
        st.markdown("Logistic Regression")
        st.markdown("**Dataset**")
        st.markdown("LoanApproval.csv")
        st.markdown("**Dataset records**")
        st.markdown(f"{record_count:,}")
        st.markdown("**Number of input features**")
        st.markdown(str(len(FEATURES)))
        st.markdown("**Test Accuracy**")
        st.markdown(f"{accuracy:.2%}")
    st.caption("The application excludes loan_id from all model inputs.")

    with st.expander("Model Information", icon=":material/model_training:"):
        st.markdown("**Model:** Logistic Regression")
        st.markdown(f"**Accuracy:** {accuracy:.2%}")
        st.markdown(f"**Precision:** {precision:.2%}")
        st.markdown(f"**Recall:** {recall:.2%}")
        st.markdown(f"**F1 Score:** {f1_score:.2%}")

st.header("Applicant Details", icon=":material/person:")

with st.form("loan_prediction_form", border=True):
    left, right = st.columns(2, gap="large")
    with left:
        no_of_dependents = st.number_input(
            "Number of dependents",
            min_value=0,
            max_value=20,
            step=1,
            key="no_of_dependents",
        )
        education = st.selectbox(
            "Education", ["Graduate", "Not Graduate"], key="education"
        )
        self_employed = st.selectbox(
            "Self employed", ["Yes", "No"], key="self_employed"
        )
        income_annum = st.number_input(
            "Annual income (₹)",
            min_value=0,
            step=100_000,
            format="%d",
            key="income_annum",
        )
        loan_amount = st.number_input(
            "Loan amount (₹)",
            min_value=0,
            step=100_000,
            format="%d",
            key="loan_amount",
        )
        loan_term = st.number_input(
            "Loan term (years)",
            min_value=1,
            max_value=50,
            step=1,
            key="loan_term",
        )
    with right:
        credit_score = st.number_input(
            "Credit Score / CIBIL Score",
            min_value=300,
            max_value=900,
            step=1,
            key="credit_score",
            help="Enter a score from 300 to 900.",
        )
        residential_assets_value = st.number_input(
            "Residential asset value (₹)",
            min_value=0,
            step=100_000,
            format="%d",
            key="residential_assets_value",
        )
        commercial_assets_value = st.number_input(
            "Commercial asset value (₹)",
            min_value=0,
            step=100_000,
            format="%d",
            key="commercial_assets_value",
        )
        luxury_assets_value = st.number_input(
            "Luxury asset value (₹)",
            min_value=0,
            step=100_000,
            format="%d",
            key="luxury_assets_value",
        )
        bank_asset_value = st.number_input(
            "Bank asset value (₹)",
            min_value=0,
            step=100_000,
            format="%d",
            key="bank_asset_value",
        )

    button_left, button_right = st.columns([3, 1])
    with button_left:
        submitted = st.form_submit_button(
            "🔮 Predict Loan Status", type="primary", width="stretch"
        )
    with button_right:
        st.form_submit_button(
            "Reset",
            icon=":material/restart_alt:",
            width="stretch",
            on_click=reset_form,
        )

st.caption("Change any applicant details, then click Predict Loan Status to calculate a new result.")

if submitted:
    encoded_values = {
        "no_of_dependents": int(no_of_dependents),
        "education": 1 if education == "Graduate" else 0,
        "self_employed": 1 if self_employed == "Yes" else 0,
        "income_annum": int(income_annum),
        "loan_amount": int(loan_amount),
        "loan_term": int(loan_term),
        "credit_score": int(credit_score),
        "residential_assets_value": int(residential_assets_value),
        "commercial_assets_value": int(commercial_assets_value),
        "luxury_assets_value": int(luxury_assets_value),
        "bank_asset_value": int(bank_asset_value),
    }
    validation_errors = validate(encoded_values)

    if validation_errors:
        for message in validation_errors:
            st.error(message, icon=":material/error:")
    else:
        try:
            prediction, probability = predict_applicant(
                bundle["model"], encoded_values
            )
        except (ValueError, TypeError):
            st.error(
                "The prediction could not be calculated. Please review the entered "
                "values and try again.",
                icon=":material/error:",
            )
        else:
            approved = prediction == 1
            status = "APPROVED" if approved else "REJECTED"
            if approved:
                st.success("✅ LOAN APPROVED", icon=":material/check_circle:")
            else:
                st.error("❌ LOAN REJECTED", icon=":material/cancel:")

            st.header("Prediction Result", icon=":material/fact_check:")
            with st.container(border=True):
                metric_cols = st.columns(4)
                metric_cols[0].metric("Predicted Status", status.title())
                metric_cols[1].metric("Model", "Logistic Regression")
                metric_cols[2].metric("Model Test Accuracy", f"{accuracy:.2%}")
                metric_cols[3].metric(
                    "Model Predicted Probability", f"{probability:.2%}"
                )
                st.caption(
                    "The displayed probability is the model's confidence in the "
                    "predicted class, not a guaranteed chance of receiving a loan."
                )

            display_details = pd.DataFrame(
                {
                    "Applicant detail": [
                        "Number of dependents",
                        "Education",
                        "Self employed",
                        "Annual income",
                        "Loan amount",
                        "Loan term",
                        "Credit score",
                        "Residential asset value",
                        "Commercial asset value",
                        "Luxury asset value",
                        "Bank asset value",
                    ],
                    "Entered value": [
                        str(no_of_dependents),
                        education,
                        self_employed,
                        rupees(income_annum),
                        rupees(loan_amount),
                        f"{loan_term} years",
                        str(credit_score),
                        rupees(residential_assets_value),
                        rupees(commercial_assets_value),
                        rupees(luxury_assets_value),
                        rupees(bank_asset_value),
                    ],
                }
            )
            st.subheader("Entered applicant details", icon=":material/table_chart:")
            st.dataframe(display_details, hide_index=True, width="stretch")

with st.expander("How the Prediction Works", icon=":material/info:"):
    st.markdown(
        """
1. The user enters the applicant's details.
2. Education and self-employment values are encoded using the project mappings.
3. The 11 values are placed in the exact training feature order and passed to the Logistic Regression model.
4. The model predicts **Approved** or **Rejected**.
5. The model's predicted-class probability is displayed with the result.
"""
    )

st.warning(
    "This application is an academic machine-learning project and should not be "
    "used as the sole basis for real financial decisions.",
    icon=":material/warning:",
)
