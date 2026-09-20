from pathlib import Path

from streamlit.testing.v1 import AppTest

from loan_logic import FEATURES, applicant_frame, predict_applicant, train_model


BASE_DIR = Path(__file__).resolve().parent


def test_dataset_pipeline_and_feature_order():
    bundle = train_model(BASE_DIR / "LoanApproval.csv")
    assert bundle["record_count"] == 4269
    assert bundle["feature_names"] == FEATURES
    assert list(bundle["model"].feature_names_in_) == FEATURES
    assert round(bundle["metrics"]["accuracy"] * 100, 2) == 80.33
    assert bundle["metrics"]["confusion_matrix"] == [[196, 127], [41, 490]]


def test_model_can_produce_both_outcomes():
    bundle = train_model(BASE_DIR / "LoanApproval.csv")
    model = bundle["model"]
    approved = {
        "no_of_dependents": 2,
        "education": 1,
        "self_employed": 0,
        "income_annum": 9_600_000,
        "loan_amount": 29_900_000,
        "loan_term": 12,
        "credit_score": 778,
        "residential_assets_value": 2_400_000,
        "commercial_assets_value": 17_600_000,
        "luxury_assets_value": 22_700_000,
        "bank_asset_value": 8_000_000,
    }
    rejected = {
        "no_of_dependents": 3,
        "education": 1,
        "self_employed": 0,
        "income_annum": 9_100_000,
        "loan_amount": 29_700_000,
        "loan_term": 20,
        "credit_score": 506,
        "residential_assets_value": 7_100_000,
        "commercial_assets_value": 4_500_000,
        "luxury_assets_value": 33_300_000,
        "bank_asset_value": 12_800_000,
    }

    assert list(applicant_frame(approved).columns) == FEATURES
    approved_status, approved_probability = predict_applicant(model, approved)
    rejected_status, rejected_probability = predict_applicant(model, rejected)
    assert approved_status == 1
    assert rejected_status == 0
    assert 0.0 <= approved_probability <= 1.0
    assert 0.0 <= rejected_probability <= 1.0
    assert approved_probability != rejected_probability


def test_streamlit_app_loads_and_predicts():
    app = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert app.success or app.error


def test_reset_button_restores_defaults():
    app = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30).run()
    app.number_input(key="credit_score").set_value(400)
    app.button[1].click().run()
    assert app.number_input(key="credit_score").value == 650
    assert app.number_input(key="loan_amount").value == 15_000_000
    assert not app.exception
