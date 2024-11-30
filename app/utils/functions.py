import pandas as pd
import joblib
from sklearn.base import ClassifierMixin, TransformerMixin


def load_artifact(filepath: str) -> ClassifierMixin | TransformerMixin:
    """Load an artifact from a file.

    Args:
        filepath (str): The path to the file.

    Returns:
        ClassifierMixin | TransformerMixin: The artifact.
    """
    return joblib.load(filepath)


def check_feature_value(
    feature: str, value: str, type: str, expected_values: list
) -> None:
    """Check if a feature contains an expected value.

    Args:
        feature (str): The feature name.
        value (str): The feature value.
        type (str): The expected type.
        expected_values (list): The expected values.

    Raises:
        ValueError: If the feature contains an unexpected value.
    """
    if value is None:
        raise ValueError(f"Missing value for feature '{feature}'")

    if type == "int":
        try:
            value = int(value)
        except ValueError:
            raise ValueError(
                f"Invalid value '{value}' for feature '{feature}'. Expected an integer."
            )

    elif type == "float":
        try:
            value = float(value)
        except ValueError:
            raise ValueError(
                f"Invalid value '{value}' for feature '{feature}'. Expected a float."
            )

    elif type == "str":
        if not isinstance(value, str):
            raise ValueError(
                f"Invalid value '{value}' for feature '{feature}'. Expected a string."
            )

    if expected_values is not None and value not in expected_values:
        raise ValueError(
            f"Unexpected value '{value}' for feature '{feature}'. "
            f"Expected one of: {', '.join(map(str, expected_values))}"
        )


def extract_features(data: dict) -> pd.DataFrame:
    """Extract features from the input data.

    Args:
        data (dict): The input data.
        encoders (dict): The encoders to use.

    Returns:
        DataFrame: The extracted features.

    Raises:
        ValueError: If any required feature is missing.
    """
    expected_keys = {
        "Lead Origin": [
            "str",
            ["API", "Landing Page Submission", "Lead Add Form", "Lead Import"],
        ],
        "Lead Source": [
            "str",
            [
                "Direct Traffic",
                "Google",
                "Olark Chat",
                "Organic Search",
                "Other",
                "Reference",
                "Referral Sites",
                "Welingak Website",
            ],
        ],
        "Do Not Email": ["int", [0, 1]],
        "TotalVisits": ["float", None],
        "Total Time Spent on Website": ["int", None],
        "Last Activity": [
            "str",
            [
                "Converted to Lead",
                "Email Bounced",
                "Email Link Clicked",
                "Email Opened",
                "Form Submitted on Website",
                "Olark Chat Conversation",
                "Other",
                "Page Visited on Website",
                "SMS Sent",
                "Unreachable",
            ],
        ],
        "Through Recommendations": ["int", [0, 1]],
        "A free copy of Mastering The Interview": ["int", [0, 1]],
        "Last Notable Activity": [
            "str",
            [
                "Email Link Clicked",
                "Email Opened",
                "Modified",
                "Olark Chat Conversation",
                "Other",
                "Page Visited on Website",
                "SMS Sent",
            ],
        ],
    }
    features = []

    unexpected_keys = set(data.keys()) - set(expected_keys)
    if unexpected_keys:
        raise ValueError(f"Unexpected key(s): {', '.join(unexpected_keys)}")

    for key, (type, expected_values) in expected_keys.items():
        value = data.get(key)
        check_feature_value(key, value, type, expected_values)
        features.append(value)

    return pd.DataFrame([features], columns=expected_keys)
