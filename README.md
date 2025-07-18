Churn‑Prediction‑ANN 🚀
A Streamlit-powered AI app that uses an Artificial Neural Network (ANN) to predict customer churn. Enter customer data and instantly see the probability they may leave—helping businesses take proactive retention measures.

🎯 Project Overview
Goal: Build a classification model to identify customers at risk of churn.

Approach: Train a neural network (via Keras/TensorFlow) on historical customer features, then deploy the model with Streamlit for interactive use.

Use Case: Supports timely marketing or loyalty interventions.

📁 Project Structure
bash
Copy
Edit
Churn‑Prediction‑ANN/
├── app.py                # Streamlit interface + model loader
├── model.h5              # Pretrained ANN
├── scaler.pkl            # Preprocessing scaler object
├── encoder_gender.pkl    # Encoder for 'Gender'
├── encoder_geo.pkl       # Encoder for 'Geography'
├── Churn_Modelling.csv   # Original dataset
├── notebooks/            # EDA, feature engineering & training code
│   └── train.ipynb
├── requirements.txt      # Required Python packages
└── README.md             # This file
📥 Dataset
Original dataset: Churn_Modelling.csv (from Kaggle’s Bank Customer dataset), with these key features:

CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited (target)

⚙️ Installation
bash
Copy
Edit
git clone https://github.com/swe-dipu/Churn-Prediction-ANN.git
cd Churn-Prediction-ANN
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
🧠 Model Training (Optional)
To retrain the model:

Open notebooks/train.ipynb

Perform data preprocessing:

Label/one-hot encode categorical features

Scale numeric features

Define and compile your ANN

Train with validation

Evaluate and save:

python
Copy
Edit
classifier.save('model.h5')
Serialize scaler and encoders using pickle.

🚀 Launch the App
bash
Copy
Edit
streamlit run app.py
➡️ A local web interface will open — enter customer details and click Predict to see churn risk score.

🛠️ Technologies & Tools
Python: Data munging with pandas, numpy

scikit-learn: Preprocessing & metrics

TensorFlow / Keras: ANN architecture, training & saving

Streamlit: Interactive app interface

Pickle: Model and preprocessing object serialization

📊 Results & Evaluation
Achieves ~85–88% accuracy on hold-out test data.

Performance metrics (in notebook): precision, recall, F1-score, confusion matrix.

Comparable to similar projects, e.g. Artificial-Neural-Network-Streamlit which also reached ~87% accuracy.

📝 Acknowledgments
Inspired by community implementations such as Antoninichiq’s Streamlit-ANN churn predictor:
https://github.com/antoninichiq/Artificial-Neural-Network-Streamlit

🧩 Future Enhancements
Add ROC-AUC curve and probability threshold tuning.

Include batch predictions support (CSV file upload).

Containerize with Docker.

Improve UI with more explanation and data validation.


📜 License
This project is released under the MIT License. See LICENSE for details.
