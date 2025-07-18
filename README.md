# Churn‑Prediction‑ANN 🚀

A **Streamlit-powered AI app** that uses an Artificial Neural Network (ANN) to predict customer churn. Enter customer data and instantly see the probability they may leave—helping businesses take proactive retention measures.

---

## 🎯 Project Overview

- **Goal**: Build a classification model to identify customers at risk of churn.
- **Approach**: Train a neural network (via Keras/TensorFlow) on historical customer features, then deploy the model with Streamlit for interactive use.
- **Use Case**: Supports timely marketing or loyalty interventions.

---

## 📁 Project Structure

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
