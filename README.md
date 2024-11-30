# ml-lead-prediction-api 🚀

## 🌟 Objective  
This Flask application is designed for **customer lead prediction** based on prior data. The project encompasses:  
- Data preparation, analysis, and cleaning 🧹  
- Feature engineering and machine learning pipeline setup 🧠  
- Training and comparison of models:  
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Support Vector Classifier (SVC)  
  - Random Forest  
  - Neural Network  

The results of these models are detailed in the notebook: **`Lead_Prediction.ipynb`**, available in the root directory.

---

## 🛠️ Features  

### API Endpoints  
- `/`  
  - **Description**: Provides information about the API and a sample JSON request for prediction.  
- `/predict`  
  - **Description**: Accepts POST requests to return lead prediction results.  

### 🧪 Test Suite  
A robust test suite is included to ensure functionality:  
- Located in the **`./tests`** folder.  
- Run tests using **`pytest`**.  
- Contributions to enhance the tests are welcome! 🤝  

---

## 🐳 Docker Support  
You can build and run the app in a **Docker container** for ease of deployment. Steps to build and execute are included.  

---

## 🚀 Getting Started  

### Prerequisites  
- Python 3.8+  
- Docker (optional for containerized deployment)  

### Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/nicvlt/ml-lead-prediction-api.git  
   cd ml-lead-prediction-api  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Run the application:  
   ```bash  
   python run.py  
   ```  

### Docker (Optional)  
1. Build the container:  
   ```bash  
   docker build -t ml-lead-prediction-api .  
   ```  
2. Run the container:  
   ```bash  
   docker run -p 5000:5000 ml-lead-prediction-api  
   ```  

---

## 📂 Project Structure  
```
ml-lead-prediction-api/  
├── Lead_Prediction.ipynb     # Notebook detailing model comparisons  
├── app/                      # Flask application files  
├── tests/                    # Unit tests  
├── requirements.txt          # Dependencies  
├── Dockerfile                # Docker configuration  
└── run.py                    # Main application entry point  
```  

---  

## 📝 License  
This project is licensed under the MIT License.  

Feel free to reach out with any questions or suggestions! 😊  