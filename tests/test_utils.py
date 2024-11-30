import pytest
import joblib
import pandas as pd

from unittest.mock import MagicMock, patch

from app.utils import format_response
from app.utils import load_artifact, check_feature_value, extract_features


# FORMAT_RESPONSE TESTS
def test_format_response_default_status():
    data = {"message": "Success"}
    expected = {"status": 200, "data": data}
    result = format_response(data)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_format_response_custom_status():
    data = {"message": "Error"}
    status = 400
    expected = {"status": status, "data": data}
    result = format_response(data, status)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_format_response_empty_data():
    data = {}
    expected = {"status": 200, "data": data}
    result = format_response(data)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_format_response_none_data():
    data = None
    expected = {"status": 200, "data": data}
    result = format_response(data)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_format_response_non_dict_data():
    data = [1, 2, 3]
    expected = {"status": 200, "data": data}
    result = format_response(data)
    assert result == expected, f"Expected {expected}, but got {result}"


# LOAD_ARTIFACT TESTS
def test_load_artifact_none_filepath():
    with pytest.raises(TypeError):
        load_artifact(None)


@patch("joblib.load")
def test_load_artifact_valid_file(mock_joblib_load):
    mock_artifact = MagicMock()
    mock_joblib_load.return_value = mock_artifact

    filepath = "valid/path/to/artifact.pkl"
    result = load_artifact(filepath)

    mock_joblib_load.assert_called_once_with(filepath)
    assert result == mock_artifact, f"Expected {mock_artifact}, but got {result}"


@patch("joblib.load")
def test_load_artifact_invalid_filepath(mock_joblib_load):
    mock_joblib_load.side_effect = FileNotFoundError

    filepath = "invalid/path/to/artifact.pkl"
    with pytest.raises(FileNotFoundError):
        load_artifact(filepath)


# CHECK_FEATURE_VALUE TESTS
def test_check_feature_value_str_valid():
    """Test for categorical features: normal behavior"""
    feature = "Lead Origin"
    value = "API"
    type = "str"
    expected_values = ["API", "Landing Page Submission", "Lead Add Form", "Lead Import"]
    check_feature_value(feature, value, type, expected_values)


def test_check_feature_value_str_invalid():
    """Test for categorical features: does it accept invalid values for categorical features?"""
    feature = "Lead Origin"
    value = "Invalid Origin"
    type = "str"
    expected_values = ["API", "Landing Page Submission", "Lead Add Form", "Lead Import"]
    with pytest.raises(
        ValueError,
        match=f"Unexpected value '{value}' for feature '{feature}'. Expected one of: {', '.join(map(str, expected_values))}",
    ):
        check_feature_value(feature, value, type, expected_values)


def test_check_feature_value_int_valid():
    """Test for binary features: normal behavior"""
    feature = "Do Not Email"
    value = 1  # arleady an int
    type = "int"
    expected_values = [0, 1]
    check_feature_value(feature, value, type, expected_values)


def test_check_feature_value_int_invalid():
    """Test for binary features: does it accept not 0 or 1?"""
    feature = "Do Not Email"
    value = 23
    type = "int"
    expected_values = [0, 1]
    with pytest.raises(
        ValueError,
        match=f"Unexpected value '{value}' for feature '{feature}'. Expected one of: {', '.join(map(str, expected_values))}",
    ):
        check_feature_value(feature, value, type, expected_values)


def test_check_feature_value_cast_int_valid():
    """Test for binary features: does it cast string to int? (normal behavior)"""
    feature = "Do Not Email"
    value = "1"  # castable to int
    type = "int"
    expected_values = [0, 1]
    check_feature_value(feature, value, type, expected_values)


def test_check_feature_value_cast_int_invalid():
    """Test for binary features: does it cast string to int? (invalid value)"""
    feature = "Do Not Email"
    value = "invalid"  # uncastable to int
    type = "int"
    expected_values = [0, 1]
    with pytest.raises(
        ValueError,
        match=f"Invalid value '{value}' for feature '{feature}'. Expected an integer.",
    ):
        check_feature_value(feature, value, type, expected_values)


def test_check_feature_value_float_valid():
    """Test for continuous features: normal behavior"""
    feature = "TotalVisits"
    value = 10.0
    type = "float"
    expected_values = None
    check_feature_value(feature, value, type, expected_values)


def test_check_feature_value_cast_float_valid():
    """Test for continuous features: does it cast string to float? (normal behavior)"""
    feature = "TotalVisits"
    value = "15.5"  # castable to float
    type = "float"
    expected_values = None
    check_feature_value(feature, value, type, expected_values)


def test_check_feature_value_cast_float_invalid():
    """Test for continuous features: does it cast string to float? (invalid value)"""
    feature = "TotalVisits"
    value = "invalid"  # uncastable to float
    type = "float"
    expected_values = None
    with pytest.raises(
        ValueError,
        match=f"Invalid value '{value}' for feature '{feature}'. Expected a float.",
    ):
        check_feature_value(feature, value, type, expected_values)


def test_check_feature_value_none_value():
    """Test for missing value parameter: normal behavior (we can call the function with None as value but that will raise an error)"""
    feature = "Lead Source"
    value = None
    type = "str"
    expected_values = [
        "Direct Traffic",
        "Google",
        "Olark Chat",
        "Organic Search",
        "Other",
        "Reference",
        "Referral Sites",
        "Welingak Website",
    ]
    with pytest.raises(ValueError, match=f"Missing value for feature '{feature}'"):
        check_feature_value(feature, value, type, expected_values)


def test_check_feature_value_none_expected_values_valid():
    """Test for missing expected values parameter: normal behavior (some features don't have expected values)"""
    feature = "Total Time Spent on Website"
    value = "30"
    type = "int"
    expected_values = None
    check_feature_value(feature, value, type, expected_values)


# EXTRACT_FEATURES TESTS
def test_extract_features_valid_data():
    """Test for normal behavior"""
    data = {
        "Lead Origin": "API",
        "Lead Source": "Google",
        "Do Not Email": 1,
        "TotalVisits": 15.5,
        "Total Time Spent on Website": 120,
        "Last Activity": "Email Opened",
        "Through Recommendations": 1,
        "A free copy of Mastering The Interview": 0,
        "Last Notable Activity": "Email Opened",
    }
    df = extract_features(data)
    assert isinstance(df, pd.DataFrame), "Expected output to be a pandas DataFrame."
    assert df.shape == (1, 9), "Expected DataFrame to have 1 row and 9 columns."
    assert list(df.columns) == [
        "Lead Origin",
        "Lead Source",
        "Do Not Email",
        "TotalVisits",
        "Total Time Spent on Website",
        "Last Activity",
        "Through Recommendations",
        "A free copy of Mastering The Interview",
        "Last Notable Activity",
    ], "Expected columns to match the feature keys."


def test_extract_features_unexpected_key():
    """Test for unexpected key in data"""
    data = {
        "Lead Origin": "API",
        "Lead Source": "Google",
        "Do Not Email": 1,
        "TotalVisits": 15.5,
        "Total Time Spent on Website": 120,
        "Last Activity": "Email Opened",
        "Through Recommendations": 1,
        "A free copy of Mastering The Interview": 0,
        "Last Notable Activity": "Email Opened",
        "Unexpected Key": "Invalid",
    }
    with pytest.raises(ValueError, match="Unknown features: Unexpected Key"):
        extract_features(data)


def test_extract_features_missing_key():
    """Test for missing key in data"""
    data = {
        "Lead Origin": "API",
        "Lead Source": "Google",
        "Do Not Email": 1,
        "TotalVisits": 15.5,
        "Total Time Spent on Website": 120,
        # "Last Activity" key is missing
        "Through Recommendations": 1,
        "A free copy of Mastering The Interview": 0,
        "Last Notable Activity": "Email Opened",
    }
    with pytest.raises(ValueError, match="Missing value for feature 'Last Activity'"):
        extract_features(data)


def test_extract_features_invalid_value_type():
    """Test for invalid value type for a feature"""
    data = {
        "Lead Origin": "API",
        "Lead Source": "Google",
        "Do Not Email": "string_instead_of_int",  # Invalid value type for 'Do Not Email'
        "TotalVisits": 15.5,
        "Total Time Spent on Website": 120,
        "Last Activity": "Email Opened",
        "Through Recommendations": 1,
        "A free copy of Mastering The Interview": 0,
        "Last Notable Activity": "Email Opened",
    }
    with pytest.raises(
        ValueError,
        match="Invalid value 'string_instead_of_int' for feature 'Do Not Email'. Expected an integer.",
    ):
        extract_features(data)


def test_extract_features_none_value():
    """Test for missing value for a feature"""
    data = {
        "Lead Origin": "API",
        "Lead Source": "Google",
        "Do Not Email": 1,
        "TotalVisits": 15.5,
        "Total Time Spent on Website": 120,
        "Last Activity": "Email Opened",
        "Through Recommendations": None,
        "A free copy of Mastering The Interview": 0,
        "Last Notable Activity": "Email Opened",
    }
    with pytest.raises(
        ValueError, match="Missing value for feature 'Through Recommendations'"
    ):
        extract_features(data)
