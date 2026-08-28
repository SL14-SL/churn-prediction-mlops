import pandas as pd
import pytest

from src.data.features.build_features import build_features


@pytest.fixture
def training_config() -> dict:
    return {
        "data": {
            "target_column": "Churn",
            "time_column": None,
            "id_columns": ["customerID"],
        },
        "features": {
            "enabled_steps": [
                "clean_names",
                "cast_numeric_types",
                "add_churn_domain_features",
                "encode_categoricals",
                "drop_configured",
                "cast_ohe_to_bool",
            ],
            "cast_to_numeric": ["totalcharges"],
            "numeric_columns": [
                "tenure", 
                "monthlycharges", 
                "totalcharges",
                "is_new_customer",
            ],
            "categorical_columns": [
                "gender",
                "seniorcitizen",
                "partner",
                "dependents",
                "phoneservice",
                "multiplelines",
                "internetservice",
                "onlinesecurity",
                "onlinebackup",
                "deviceprotection",
                "techsupport",
                "streamingtv",
                "streamingmovies",
                "contract",
                "paperlessbilling",
                "paymentmethod",
                "tenure_group",
            ],
            "drop_columns": ["customerid"],
        },
    }


@pytest.fixture
def churn_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customerID": "1234-ABCDE",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": "845.50",
                "Churn": "Yes",
            },
            {
                "customerID": "5678-FGHIJ",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 45.0,
                "TotalCharges": "135.0",
                "Churn": "No",
            },
        ]
    )


def test_build_features_churn_pipeline(training_config, churn_df):
    result = build_features(churn_df, config=training_config)

    assert len(result) == 2
    assert "customerid" not in result.columns

    assert "totalcharges" in result.columns
    assert pd.api.types.is_numeric_dtype(result["totalcharges"])

    assert "gender_Male" in result.columns
    assert "partner_Yes" in result.columns
    assert "contract_One year" in result.columns
    assert "paymentmethod_Mailed check" in result.columns

    assert pd.api.types.is_bool_dtype(result["gender_Male"])
    assert "is_new_customer" in result.columns

    assert (
        result.loc[
            0,
            "is_new_customer",
        ]
        == 1
    )

    assert (
        result.loc[
            1,
            "is_new_customer",
        ]
        == 1
    )

    assert any(
        column.startswith(
            "tenure_group_"
        )
        for column in result.columns
    )



def test_build_features_without_target_for_inference_like_payload(training_config, churn_df):
    inference_df = churn_df.drop(columns=["Churn"])

    result = build_features(inference_df, config=training_config)

    assert len(result) == 2
    assert "customerid" not in result.columns
    assert "churn" not in result.columns
    assert "totalcharges" in result.columns
    assert pd.api.types.is_numeric_dtype(result["totalcharges"])


def test_build_features_handles_empty_dataframe(training_config):
    df = pd.DataFrame(
        columns=[
            "customerID",
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
        ]
    )

    result = build_features(df, config=training_config)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_build_features_unknown_step_raises(training_config, churn_df):
    cfg = {
        **training_config,
        "features": {
            **training_config["features"],
            "enabled_steps": ["clean_names", "unknown_step"],
        },
    }

    with pytest.raises(ValueError, match="Unknown feature step"):
        build_features(churn_df, config=cfg)


def test_build_features_respects_enabled_steps(training_config, churn_df):
    cfg = {
        "data": training_config["data"],
        "features": {
            "enabled_steps": ["clean_names", "cast_numeric_types"],
            "cast_to_numeric": ["totalcharges"],
            "drop_columns": [],
        },
    }

    result = build_features(churn_df, config=cfg)

    assert "customerid" in result.columns
    assert "totalcharges" in result.columns
    assert pd.api.types.is_numeric_dtype(result["totalcharges"])
    assert "gender_Male" not in result.columns