# 🎓 Student Marks Predictor - College ML Project

An interactive Machine Learning web application built using **Python**, **Scikit-Learn**, and **Streamlit** to forecast a student's final examination scores. This project helps identify how different student habits (study time, sleep, attendance) and past academic success affect their final grade.

---

## 🌟 Project Highlights
* **Interactive Predictions**: Real-time evaluation of final exam marks through intuitive sliders.
* **Optimized Model Training**: Preprocesses student statistics and compares **Linear Regression** and **Random Forest Regressor** to choose the most accurate model.
* **Exploratory Data Analysis (EDA)**: Interactive data visualizations showing heatmaps of feature correlations and linear trends.
* **Habit Optimization Scenario**: A "What-If" helper that calculates exactly how many hours of weekly study a student needs to achieve their target exam score.
* **Professional Layout**: Curated modern styling, custom Google Fonts (`Outfit`), and smooth gradient cards.

---

## 🛠️ Technology Stack
* **Language**: Python 3.11+
* **ML Library**: Scikit-Learn (Linear Regression, Random Forest, StandardScaler)
* **Data Processing**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Web UI Framework**: Streamlit

---

## 📁 Project Directory Structure
```text
ML Project - College/
├── .venv/                      # Python virtual environment
├── student_exam_scores.csv     # Dataset containing student academic records
├── requirements.txt            # Project dependencies list
├── train.py                    # Script to preprocess dataset and train models
├── app.py                      # Interactive Streamlit application code
├── model.pkl                   # Serialized final trained model
├── scaler.pkl                  # Serialized feature standard scaler
├── model_metadata.pkl          # Performance metrics (MAE, R2 score) metadata
└── README.md                   # Project documentation
```

---

## 🚀 Setup & Execution Guide

Follow these simple steps to set up and run the application on your computer:

### 1. Set Up the Virtual Environment & Install Dependencies
Make sure you are in the project folder in your terminal, then run:
```bash
# Verify pip is in the virtual environment and install dependencies
.venv/bin/pip install -r requirements.txt
```

### 2. Train the Machine Learning Model
Before running the Streamlit app, we must train the model and export the weights:
```bash
.venv/bin/python train.py
```
*This will print evaluation scores comparing Linear Regression and Random Forest on the terminal and output `model.pkl`, `scaler.pkl`, and `model_metadata.pkl`.*

### 3. Run the Web Application
Launch the interactive web portal:
```bash
.venv/bin/streamlit run app.py
```
*Once launched, the terminal will provide a Local URL (usually `http://localhost:8501`) that will automatically open in your web browser!*

---

## 📊 Dataset & Features
The project trains on `student_exam_scores.csv`, containing columns representing student profiles:
1. `hours_studied` (Float): Average hours spent studying per week.
2. `sleep_hours` (Float): Average daily sleep duration.
3. `attendance_percent` (Float): Percentage of classes attended.
4. `previous_scores` (Integer): Score in previous mid-terms (out of 100).
5. `exam_score` (Target - Float): Final exam score to predict.

### Model Evaluation Results
* **Linear Regression**:
  - **Mean Absolute Error (MAE)**: ~2.31 marks
  - **R-squared ($R^2$) Score**: ~85.4% (Explains 85% of variance)
* **Random Forest Regressor**:
  - **Mean Absolute Error (MAE)**: ~2.95 marks
  - **R-squared ($R^2$) Score**: ~79.5%

*Linear Regression performs better on this dataset and has been selected as the primary production engine for transparent model coefficient analysis.*
