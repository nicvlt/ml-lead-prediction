import pytest
import pandas as pd
from app.services import predict


def test_predict_valid():
    """Test the predict function with valid input data."""
    data = {
        "Lead Origin": ["Lead Add Form"],
        "Lead Source": ["Google"],
        "Do Not Email": ["1"],
        "TotalVisits": [5.0],
        "Total Time Spent on Website": [456],
        "Last Activity": ["Email Opened"],
        "Through Recommendations": ["0"],
        "A free copy of Mastering The Interview": ["1"],
        "Last Notable Activity": ["SMS Sent"],
    }

    df = pd.DataFrame(data)

    prediction = predict(df)

    assert prediction == "Not Converted"


def test_predict_unseen_labels():
    """Test the predict function with unseen labels."""
    data = {
        "Lead Origin": ["Bad value"],
        "Lead Source": ["Google"],
        "Do Not Email": ["0"],
        "TotalVisits": [5.0],
        "Total Time Spent on Website": [456],
        "Last Activity": ["Email Opened"],
        "Through Recommendations": ["0"],
        "A free copy of Mastering The Interview": ["1"],
        "Last Notable Activity": ["SMS Sent"],
    }

    df = pd.DataFrame(data)

    with pytest.raises(ValueError) as exc:
        predict(df)

    assert str(exc.value) == "y contains previously unseen labels: 'Bad value'"
