from flask import Blueprint, request, jsonify
from app.services.ml_service import predict
from app.utils.helpers import format_response
from app.utils.functions import extract_features
from app.utils.constants import WELCOME_MESSAGE

bp = Blueprint("routes", __name__)


@bp.route("/")
def home():
    """Home route"""
    return WELCOME_MESSAGE

@bp.route('/registrazione', methods=['GET', 'POST'])

@bp.route("/predict", methods=["POST"])
def predict_route():
    """Route to handle model predictions"""
    try:
        data = request.get_json()
        if not data:
            return format_response({"error": "No data provided"}, status=400)

        try:
            input_features = extract_features(data)
        except ValueError as e:
            return format_response({"error": str(e)}, status=400)

        prediction = predict(input_features)

        return format_response({"prediction": prediction})

    except Exception as e:
        return format_response({"error": str(e)}, status=500)
