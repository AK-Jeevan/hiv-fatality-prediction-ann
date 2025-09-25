# 🧠 HIV/AIDS Fatality Prediction using ANN

This project uses an Artificial Neural Network (ANN) to predict annual HIV/AIDS-related fatalities based on global cause-of-death data.

---

## 🗂️ Dataset Overview

- **Source**: [Global Cause of Death Dataset – Kaggle or WHO](https://www.who.int/data/gho/data/indicators/indicator-details/GHO/number-of-deaths-due-to-hiv-aids)
- **Features**: Various numeric indicators of mortality causes (excluding HIV)
- **Target**: Annual HIV/AIDS-related fatalities
- **Preprocessing**:
  - Dropped non-numeric columns (`Entity`, `Code`, `Year`)
  - Target column selected via keyword match (`"HIV"`)

---

## 🧠 Model Architecture

A deep ANN built with Keras and TensorFlow:

- Input: Multiple mortality-related features
- Hidden Layers:
  - Dense(128) → BatchNormalization → Dropout(0.3)
  - Dense(64) → BatchNormalization → Dropout(0.2)
  - Dense(32)
- Output: Dense(1) for regression
- Optimizer: Adam
- Loss Function: Mean Absolute Error (MAE)
- Metrics: Mean Squared Error (MSE)
- Regularization: Dropout + EarlyStopping

---

## 🔄 Workflow

1. **Data Cleaning**: Remove duplicates and missing values
2. **Feature Selection**: Drop non-numeric columns and isolate HIV fatality column
3. **Train-Test Split**: 80/20
4. **Scaling**: RobustScaler for outlier resistance
5. **Model Training**: 200 epochs with validation split and early stopping
6. **Evaluation**: MAE, MSE, and R² score
7. **Visualization**:
   - Training history (loss vs. epochs)
   - Actual vs. predicted scatter plot

---

## 📊 Results

- **Test MAE**: ~`[Insert your result here]`
- **Test MSE**: ~`[Insert your result here]`
- **R² Score**: ~`[Insert your result here]`

---

## 🚀 How to Run

### 🧰 Requirements
pip install numpy pandas matplotlib scikit-learn tensorflow keras

## 📜 License
This project is licensed under the MIT License.

