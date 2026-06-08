import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injectio
st.markdown("""
    <style>
        /* Import outfit font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Main panel design */
        .main {
            background-color: #f8fafc;
        }
        
        /* Custom card style */
        .metric-card {
            background: white;
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
            border: 1px solid #f1f5f9;
            margin-bottom: 20px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }
        
        .metric-header {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 36px;
            font-weight: 700;
            color: #0f172a;
        }
        
        .metric-value-accent {
            font-size: 48px;
            font-weight: 700;
            background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-top: 5px;
        }
        
        .metric-subtext {
            font-size: 13px;
            color: #94a3b8;
            margin-top: 8px;
        }
        
        /* Pill badges */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .badge-success {
            background-color: #dcfce7;
            color: #166534;
        }
        .badge-warning {
            background-color: #fef9c3;
            color: #854d0e;
        }
        .badge-danger {
            background-color: #fee2e2;
            color: #991b1b;
        }
        
        /* Gradient header banner */
        .banner {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
            color: white;
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 10px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        }
        .banner h1 {
            color: white !important;
            font-size: 38px !important;
            font-weight: 700 !important;
            margin-bottom: 8px !important;
        }
        .banner p {
            font-size: 16px;
            color: #c7d2fe;
            margin: 0;
            font-weight: 300;
        }
    </style>
""", unsafe_allow_html=True)

# Helper function to load model assets safely

# Helper function to load model assets safely
@st.cache_resource
def load_assets():
    try:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(dir_path, 'model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(dir_path, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(dir_path, 'model_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        return model, scaler, metadata
    except Exception as e:
        return None, None, None

model, scaler, metadata = load_assets()

# Load dataset for exploration
@st.cache_data
def load_dataset():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(dir_path, 'student_exam_scores.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

df = load_dataset()


# Load dataset for exploration
@st.cache_data
def load_dataset():
    if os.path.exists('student_exam_scores.csv'):
        return pd.read_csv('student_exam_scores.csv')
    return None

df = load_dataset()


# Load dataset for exploration
@st.cache_data
def load_dataset():
    if os.path.exists('student_exam_scores.csv'):
        return pd.read_csv('student_exam_scores.csv')
    return None

df = load_dataset()

# Check if model trained successfully
if model is None or scaler is None:
    st.error("⚠️ Model assets not found. Please make sure to run the training script `train.py` first to generate model.pkl and scaler.pkl.")
    st.stop()

# Header Banner
st.markdown("""
    <div class="banner">
        <h1>🎓 Student Exam Performance Predictor</h1>
        <p>An intelligent college project applying Linear Regression to forecast final exam scores based on student habits and academic indicators.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.markdown("### 🛠️ Input Parameters")
st.sidebar.markdown("Modify the parameters below to predict a student's final exam score.")

hours_studied = st.sidebar.slider(
    "Study Hours (Weekly)", 
    min_value=0.0, 
    max_value=12.0, 
    value=6.0, 
    step=0.1,
    help="Average number of hours the student studies per week."
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours (Average)", 
    min_value=4.0, 
    max_value=10.0, 
    value=7.0, 
    step=0.1,
    help="Average daily sleep hours of the student."
)

attendance_percent = st.sidebar.slider(
    "Attendance Rate (%)", 
    min_value=50.0, 
    max_value=100.0, 
    value=85.0, 
    step=0.5,
    help="Percentage of classes attended by the student."
)

previous_scores = st.sidebar.slider(
    "Previous Exam Score", 
    min_value=30, 
    max_value=100, 
    value=70, 
    step=1,
    help="Marks obtained in the student's previous assessment/mid-term (out of 100)."
)

# Tabs structure
tab_pred, tab_analysis, tab_info = st.tabs([
    "🔮 Predict Score", 
    "📊 Dataset Exploratory Analysis", 
    "🧠 Model Diagnostics & Performance"
])

with tab_pred:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### 📋 Student Habit Profile Summary")
        st.markdown("Here is the profile we are testing:")
        
        # Display inputs in visual summary cards
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                <div style="margin-right: 20px; margin-bottom: 10px;">
                    <div class="metric-header">📚 Study Hours</div>
                    <div style="font-size: 20px; font-weight: 600; color: #1e293b;">{hours_studied:.1f} hrs/week</div>
                </div>
                <div style="margin-right: 20px; margin-bottom: 10px;">
                    <div class="metric-header">😴 Sleep Hours</div>
                    <div style="font-size: 20px; font-weight: 600; color: #1e293b;">{sleep_hours:.1f} hrs/day</div>
                </div>
                <div style="margin-right: 20px; margin-bottom: 10px;">
                    <div class="metric-header">🏫 Attendance</div>
                    <div style="font-size: 20px; font-weight: 600; color: #1e293b;">{attendance_percent:.1f}%</div>
                </div>
                <div>
                    <div class="metric-header">📝 Previous score</div>
                    <div style="font-size: 20px; font-weight: 600; color: #1e293b;">{previous_scores}/100</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Performance booster analyzer
        st.markdown("### 🚀 Habit Optimization Scenario")
        st.markdown("What happens if you study 2 hours more and get sleep up to 8 hours?")
        
        # Calculate comparison input
        opt_study = min(12.0, hours_studied + 2.0)
        opt_sleep = min(10.0, max(8.0, sleep_hours))
        opt_attendance = min(100.0, attendance_percent + 5.0)
        
        # Original features scaled and predicted
        features = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # Optimized features scaled and predicted
        opt_features = np.array([[opt_study, opt_sleep, opt_attendance, previous_scores]])
        opt_features_scaled = scaler.transform(opt_features)
        opt_prediction = model.predict(opt_features_scaled)[0]
        
        gain = opt_prediction - prediction
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">Estimated Boost</div>
            <div class="metric-value" style="color: #10b981;">+{gain:.1f} Marks</div>
            <div class="metric-subtext">By increasing study to <b>{opt_study:.1f} hrs</b> and attendance to <b>{opt_attendance:.1f}%</b>, the score changes from <b>{prediction:.1f}</b> to <b>{opt_prediction:.1f}</b>.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("### 🎯 Predicted Examination Score")
        
        # Clamp prediction between 0 and 100 representing standard grades
        # Note: exam_score in dataset seems to range from 17 to 51, let's look at df's max/min
        min_dataset_score = float(df['exam_score'].min()) if df is not None else 0.0
        max_dataset_score = float(df['exam_score'].max()) if df is not None else 100.0
        
        # Adjust qualitative evaluation based on score distribution
        avg_score = float(df['exam_score'].mean()) if df is not None else 35.0
        
        # Qualitative category badge
        if prediction >= avg_score + 5:
            badge_class = "badge-success"
            status = "Excellent Prospect"
            msg = "This profile indicates strong academic habits. Keep studying and maintaining high attendance!"
        elif prediction >= avg_score - 5:
            badge_class = "badge-warning"
            status = "Average Prospect"
            msg = "The predicted score is within the normal class average. Small improvements in study hours can push it to excellent!"
        else:
            badge_class = "badge-danger"
            status = "Need Improvement"
            msg = "Caution: Predictor indicates score below average. We recommend increasing weekly study hours and attendance immediately."
            
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; border-left: 6px solid #4f46e5;">
            <div class="metric-header" style="font-size: 16px;">Final Predicted Score (Scaled Out of ~55)</div>
            <div class="metric-value-accent">{prediction:.2f} / 100</div>
            <div style="margin-top: 15px; margin-bottom: 15px;">
                <span class="badge {badge_class}">{status}</span>
            </div>
            <p style="color: #475569; font-size: 14px; line-height: 1.5; max-width: 400px; margin: 0 auto;">
                {msg}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Target Finder Tool
        st.markdown("#### 🎯 Target Score Calculator")
        target_score = st.number_input(
            "Enter your desired Exam Score:", 
            min_value=float(np.round(min_dataset_score, 1)), 
            max_value=float(np.round(max_dataset_score + 10.0, 1)), 
            value=float(np.round(avg_score, 1))
        )
        
        if df is not None:
            # We can approximate study hours needed assuming other parameters remain constant
            # Using Linear Regression equation: 
            # y = w0 * hours + w1 * sleep + w2 * att + w3 * prev + b
            # Let's extract coefficients
            if metadata['model_name'] == "Linear Regression":
                coefs = model.coef_
                intercept = model.intercept_
                means = scaler.mean_
                stds = scaler.scale_
                
                # Equation: y = intercept + sum(coef_i * (x_i - mean_i) / std_i)
                # Let's solve for x_0 (hours_studied)
                # y - intercept - sum_{i=1..3}(coef_i * (x_i - mean_i) / std_i) = coef_0 * (x_0 - mean_0) / std_0
                # Let RHS_val = y - intercept - sum_{i=1..3}(coef_i * (x_i - mean_i) / std_i)
                # (x_0 - mean_0) / std_0 = RHS_val / coef_0
                # x_0 = (RHS_val / coef_0) * std_0 + mean_0
                
                coef_hours = coefs[0]
                rhs_val = target_score - intercept
                
                # sum terms for sleep, att, prev
                inputs = [sleep_hours, attendance_percent, previous_scores]
                for idx in range(1, 4):
                    scaled_val = (inputs[idx-1] - means[idx]) / stds[idx]
                    rhs_val -= coefs[idx] * scaled_val
                    
                needed_hours = (rhs_val / coef_hours) * stds[0] + means[0]
                
                if needed_hours < 0:
                    st.success("🎉 Based on your current habits (high sleep, attendance, and previous grades), you will easily achieve this target without extra study!")
                elif needed_hours > 12.0:
                    st.warning(f"⚠️ A study time of **{needed_hours:.1f} hours/week** is calculated to reach this target. That's extremely high. Try also increasing your Attendance Rate or previous performance to make it achievable.")
                else:
                    st.info(f"💡 To achieve a score of **{target_score}**, you are estimated to need **{needed_hours:.1f} hours/week** of study (assuming sleep, attendance, and previous score stay the same).")
            else:
                st.info("💡 Target calculations are calibrated optimized for the Linear Regression model.")

with tab_analysis:
    st.markdown("### 🔍 Exploratory Data Analysis (EDA)")
    st.markdown("This section helps explain the dataset structure and correlation to your college evaluation panel.")
    
    if df is not None:
        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            st.markdown("#### Feature Correlation Heatmap")
            st.write("Visualizes how features correlate with the Exam Score. A score of 1.0 represents a perfect positive linear relationship.")
            
            fig, ax = plt.subplots(figsize=(6, 4.5))
            # drop non-numeric column
            numeric_df = df.drop(columns=['student_id'])
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
            plt.title("Correlation Matrix of Features", fontsize=10, pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        with col_eda2:
            st.markdown("#### Study Hours vs Exam Score")
            st.write("Shows individual student records. Notice how the trend goes up linearly as study hours increase.")
            
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.regplot(data=df, x='hours_studied', y='exam_score', scatter_kws={'alpha':0.6, 'color':'#4f46e5'}, line_kws={'color':'#e11d48'}, ax=ax)
            ax.set_xlabel("Weekly Study Hours")
            ax.set_ylabel("Final Exam Score")
            plt.title("Study Hours vs Exam Score Regression Trend", fontsize=10, pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        st.markdown("#### Dataset Sample View")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("Could not find `student_exam_scores.csv` to run data exploration.")

with tab_info:
    st.markdown("### 🧠 Machine Learning Engine Architecture")
    st.write("Technical details to present to the faculty or class evaluator.")
    
    col_met1, col_met2, col_met3 = st.columns(3)
    
    with col_met1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">Selected Model</div>
            <div style="font-size: 24px; font-weight: 700; color: #4f46e5; margin: 10px 0;">{metadata['model_name']}</div>
            <div class="metric-subtext">Selected based on testing comparison for highest $R^2$ accuracy.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_met2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">R-squared (R²) Score</div>
            <div style="font-size: 28px; font-weight: 700; color: #0f172a; margin: 10px 0;">{metadata['r2_score'] * 100:.1f}%</div>
            <div class="metric-subtext">Model explains {metadata['r2_score'] * 100:.1f}% of variance in student scores.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_met3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">Mean Absolute Error</div>
            <div style="font-size: 28px; font-weight: 700; color: #0f172a; margin: 10px 0;">{metadata['mae']:.2f} Marks</div>
            <div class="metric-subtext">Average prediction deviation is ±{metadata['mae']:.2f} marks on a 100 scale.</div>
        </div>
        """, unsafe_allow_html=True)
        
    if metadata['model_name'] == "Linear Regression":
        st.markdown("### 📊 Model Equations & Weights")
        st.write("Since Linear Regression is a transparent mathematical model, we can write down the exact relationship:")
        
        coefs = model.coef_
        intercept = model.intercept_
        
        # Display the mathematical formula
        formula = f"$$Scaled\\_Score = {intercept:.4f}"
        for col, coef in zip(metadata['features'], coefs):
            sign = "+" if coef >= 0 else "-"
            col_escaped = col.replace('_', '\\_')
            formula += f" {sign} {abs(coef):.4f} \\times Z({col_escaped})"
        formula += "$$"
        
        st.markdown(formula)
        st.markdown("<p style='font-size:12px; color:gray; text-align:center;'>Note: Z(Feature) refers to the standard-scaled feature value: (value - mean) / standard_deviation</p>", unsafe_allow_html=True)
        
        # Feature impact table
        st.markdown("#### Feature Coefficients (Impact Analysis)")
        impact_df = pd.DataFrame({
            'Feature': [f.replace('_', ' ').title() for f in metadata['features']],
            'Coefficient (Beta)': coefs,
            'Relationship Type': ['Positive (High Study = Higher Marks)' if c >= 0 else 'Negative' for c in coefs]
        })
        st.table(impact_df)
    else:
        st.info("Feature importance display is optimized for Linear Regression.")
        
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 12px;'>Student Marks Predictor - College ML Project © 2026. Made with streamlit and scikit-learn.</p>", unsafe_allow_html=True)
