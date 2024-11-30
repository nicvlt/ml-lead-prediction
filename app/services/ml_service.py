import pandas as pd
import numpy as np
from app.utils.constants import (
    MODEL_PATH,
    LAST_ACTIVITY_ENCODER_PATH,
    LAST_NOTABLE_ACTIVITY_ENCODER_PATH,
    LEAD_SOURCE_ENCODER_PATH,
    LEAD_ORIGIN_ENCODER_PATH,
    SCALER_PATH,
)
from app.utils.functions import load_artifact

model = load_artifact(MODEL_PATH)
encoders = {
    "Last Activity": load_artifact(LAST_ACTIVITY_ENCODER_PATH),
    "Last Notable Activity": load_artifact(LAST_NOTABLE_ACTIVITY_ENCODER_PATH),
    "Lead Source": load_artifact(LEAD_SOURCE_ENCODER_PATH),
    "Lead Origin": load_artifact(LEAD_ORIGIN_ENCODER_PATH),
}
scaler = load_artifact(SCALER_PATH)


def predict(input_data: pd.DataFrame) -> int:
    """Make a prediction using the model.

    Args:
        input_data (DataFrame): The input data with features.

    Returns:
        int: The class prediction.
    """
    for key in input_data.keys():
        if key in encoders.keys():  # Categorical features
            input_data[key] = encoders[key].transform(input_data[key])

    input_data = scaler.transform(input_data)

    class_prediction = model.predict(input_data)
    assert len(class_prediction) == 1
    assert class_prediction[0] in [0, 1]

    if class_prediction[0] == 0:
        return "Not Converted"
    return "Converted"
