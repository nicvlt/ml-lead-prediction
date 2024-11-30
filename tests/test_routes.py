import pytest
import json

from app.routes import bp as routes_bp
from app.utils import WELCOME_MESSAGE
from run import create_app


@pytest.fixture
def client():
    """Fixture to provide a test client."""
    app = create_app()
    with app.test_client() as client:
        yield client


def test_home_route(client):
    """Test the home route returns the correct welcome message."""
    response = client.get("/")

    assert (
        response.status_code == 200
    ), f"Expected status code 200, but got {response.status_code}"

    assert (
        response.data.decode() == WELCOME_MESSAGE
    ), f"Expected message '{WELCOME_MESSAGE}', but got {response.data.decode()}"


def test_predict_route_valid_not_converted(client):
    """Test the /predict route with valid input data."""

    input_data = {
        "Lead Origin": "Lead Add Form",
        "Lead Source": "Google",
        "Do Not Email": "1",
        "TotalVisits": 5.0,
        "Total Time Spent on Website": 456,
        "Last Activity": "Email Opened",
        "Through Recommendations": "0",
        "A free copy of Mastering The Interview": "1",
        "Last Notable Activity": "SMS Sent",
    }

    expected_response = {"data": {"prediction": "Not Converted"}, "status": 200}

    response = client.post(
        "/predict",
        data=json.dumps(input_data),
        content_type="application/json",
    )
    response_json = json.loads(response.data.decode())

    assert (
        response_json["status"] == 200
    ), f"Expected status code 200, but got {response_json.get('status')}"

    assert (
        response_json == expected_response
    ), f"Expected {expected_response}, but got {response_json}"


def test_predict_route_valid_converted(client):
    """Test the /predict route with valid input data."""
    input_data = {
        "Lead Origin": "Lead Add Form",
        "Lead Source": "Google",
        "Do Not Email": "0",
        "TotalVisits": 5.0,
        "Total Time Spent on Website": 456,
        "Last Activity": "Email Opened",
        "Through Recommendations": "0",
        "A free copy of Mastering The Interview": "1",
        "Last Notable Activity": "SMS Sent",
    }

    expected_response = {"data": {"prediction": "Converted"}, "status": 200}

    response = client.post(
        "/predict",
        data=json.dumps(input_data),
        content_type="application/json",
    )
    response_json = json.loads(response.data.decode())
    assert (
        response_json["status"] == 200
    ), f"Expected status code 200, but got {response.get('status')}"

    assert (
        response_json == expected_response
    ), f"Expected {expected_response}, but got {response_json}"


def test_predict_route_no_data(client):
    """Test the /predict route with empty JSON, expecting an error response."""
    expected_response = {"data": {"error": "No data provided"}, "status": 400}

    response = client.post(
        "/predict", data=json.dumps({}), content_type="application/json"
    )
    response_json = json.loads(response.data.decode())

    assert (
        response_json["status"] == 400
    ), f"Expected status code 400, but got {response_json.get('status')}"

    assert (
        response_json == expected_response
    ), f"Expected {expected_response}, but got {response_json}"
