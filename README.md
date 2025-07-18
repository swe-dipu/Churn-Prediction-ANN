
# Churn‑Prediction‑ANN 🚀

A **Streamlit-powered AI app** that uses an Artificial Neural Network (ANN) to predict customer churn. Enter customer data and instantly see the probability they may leave — helping businesses take proactive retention measures.

---

## 🎯 Project Overview

- **Goal**: Build a classification model to identify customers at risk of churn.
- **Approach**: Train a neural network (via Keras/TensorFlow) on historical customer features, then deploy the model with Streamlit for interactive use.
- **Use Case**: Supports timely marketing or loyalty interventions.

---

## 📁 Project Structure

```
Churn‑Prediction‑ANN/
├── app.py                # Streamlit interface + model loader
├── model.h5              # Pretrained ANN model
├── scaler.pkl            # Preprocessing scaler object
├── encoder_gender.pkl    # Encoder for 'Gender'
├── encoder_geo.pkl       # Encoder for 'Geography'
├── Churn_Modelling.csv   # Original dataset
├── notebooks/            # Jupyter notebook for EDA & model training
│   └── train.ipynb
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation (this file)
```

---

## 📥 Dataset

The dataset used is `Churn_Modelling.csv` sourced from Kaggle's **Bank Customer Churn** dataset.

Key features used in the model:

- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`

**Target Variable**: `Exited` (1 = Customer churned, 0 = Customer retained)

---

## 🧠 Model Training (Optional)

To retrain the ANN model:

1. Open `notebooks/train.ipynb`
2. Perform data preprocessing:
   - Encode categorical features (e.g., Gender, Geography)
   - Scale numerical features using StandardScaler
3. Build and compile the ANN model using Keras
4. Train the model on the dataset with validation
5. Save the trained model and preprocessing objects:
   ```python
   classifier.save('model.h5')
   pickle.dump(scaler, open('scaler.pkl', 'wb'))
   pickle.dump(encoder_gender, open('encoder_gender.pkl', 'wb'))
   pickle.dump(encoder_geo, open('encoder_geo.pkl', 'wb'))
   ```

---

## 🚀 Launch the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser.  
You can enter customer information to check churn probability.

---

## 🛠️ Technologies & Tools Used

- **Python**: Core programming language
- **pandas, numpy**: Data analysis and manipulation
- **scikit-learn**: Preprocessing, model evaluation
- **TensorFlow / Keras**: ANN modeling
- **Streamlit**: Web app framework for machine learning
- **pickle**: Model and scaler serialization

---

## 📊 Results & Model Evaluation

- Achieved **85–88% accuracy** on the validation dataset.
- Evaluation metrics covered in `train.ipynb` include:
  - Confusion Matrix
  - Precision, Recall, F1-Score
  - Accuracy Score

Model performance is comparable to similar ANN-based churn predictors.

---

## 🧩 Future Enhancements

- Add **ROC-AUC curve** visualization
- Support **batch predictions** via CSV upload
- Deploy with **Docker** for easier scalability
- Improve UI/UX with additional user feedback and validations


---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
