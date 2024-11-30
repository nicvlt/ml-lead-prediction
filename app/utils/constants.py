MODEL_PATH = "artifacts/best_svm.joblib"
LAST_ACTIVITY_ENCODER_PATH = "artifacts/last_activity_encoder.joblib"
LAST_NOTABLE_ACTIVITY_ENCODER_PATH = "artifacts/last_notable_activity_encoder.joblib"
LEAD_SOURCE_ENCODER_PATH = "artifacts/lead_source_encoder.joblib"
LEAD_ORIGIN_ENCODER_PATH = "artifacts/lead_origin_encoder.joblib"
SCALER_PATH = "artifacts/scaler.joblib"

WELCOME_MESSAGE = """
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; }
        h1 { color: #333; }
        p { margin-bottom: 15px; }
        pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; }
        .code-block { background-color: #f9f9f9; border-left: 4px solid #4CAF50; padding: 15px; }
        a { color: #4CAF50; text-decoration: none; }
    </style>
</head>
<body>
    <h1>Welcome to the Lead Conversion Prediction API</h1>
    <p>To make a prediction, send a POST request to the <code>/predict</code> endpoint with the following format:</p>
    <pre class="code-block">
{
    "Lead Origin": "Lead Add Form",
    "Lead Source": "Google",
    "Do Not Email": "0",
    "TotalVisits": 5.0,
    "Total Time Spent on Website": 456,
    "Last Activity": "Email Opened",
    "Through Recommendations": "0",
    "A free copy of Mastering The Interview": "1",
    "Last Notable Activity": "SMS Sent"
}
    </pre>
    <p>The response will be a JSON object with the prediction.</p>
    <p>For more information, visit the documentation at: <a href="https://github.com/nicvlt/ml-lead-prediction" target="_blank">https://github.com/nicvlt/ml-lead-prediction</a></p>
    <p>Enjoy!</p>
</body>
</html>
"""
