import streamlit as st
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- PAGE CONFIG ---
st.set_page_config(page_title="ML Classification Cockpit", page_icon="🔬", layout="wide")


# --- STEP 1: LOAD & PREPROCESS DATA ---
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.target_names, data.feature_names


def preprocess_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data (Critical for SVM/Logistic Regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# --- UI HEADER ---
st.title("🔬 MCert Classification Cockpit")
st.markdown(
    "Compare **Logistic Regression**, **Decision Trees**, and **SVM** on the Breast Cancer Wisconsin Diagnostic Dataset.")
st.markdown("---")

# Load Data
df, target_names, feature_names = load_data()

# --- SIDEBAR: DATA EXPLORER ---
with st.sidebar:
    st.header("🗂️ Data Control")
    if st.checkbox("Show Raw Data Sample"):
        st.dataframe(df.head(10))

    st.info(f"Total Samples: {df.shape[0]}")
    st.info(f"Features: {df.shape[1] - 1}")
    st.info(f"Classes: {list(target_names)}")

# --- MAIN COCKPIT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Model Configuration")
    model_choice = st.radio(
        "Select Algorithm:",
        ["Logistic Regression", "Decision Tree", "Support Vector Machine (SVM)"]
    )

    # NEW: Tree Depth Control (Visual clarity)
    if model_choice == "Decision Tree":
        max_depth = st.slider("Max Tree Depth (for visualization)", 1, 5, 3)

    run_btn = st.button("🚀 Train & Evaluate", type="primary")

with col2:
    st.subheader("📊 Performance Metrics")

    if run_btn:
        # 1. Prepare Data
        with st.status("🛠️ Preprocessing & Training...", expanded=True) as status:
            time.sleep(1)  # Simulated Delay
            X_train, X_test, y_train, y_test = preprocess_data(df)

            # 2. Initialize Model
            if model_choice == "Logistic Regression":
                model = LogisticRegression()
            elif model_choice == "Decision Tree":
                # Use selected depth to keep chart readable
                model = DecisionTreeClassifier(max_depth=max_depth if 'max_depth' in locals() else 3)
            else:
                model = SVC()

            # 3. Train Model
            model.fit(X_train, y_train)
            status.update(label="✅ Training Complete", state="complete", expanded=False)

        # 4. Evaluation
        y_pred = model.predict(X_test)

        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Display Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.2%}")
        m2.metric("Precision", f"{prec:.2%}")
        m3.metric("Recall", f"{rec:.2%}")
        m4.metric("F1 Score", f"{f1:.2%}")

        # 5. Visualization
        st.markdown("---")

        # NEW: Conditional Visualization
        if model_choice == "Decision Tree":
            st.subheader("🌳 Decision Tree Logic Flow")
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(model, filled=True, feature_names=feature_names, class_names=target_names, fontsize=10)
            st.pyplot(fig)
        else:
            # Standard Confusion Matrix for others
            st.subheader("📉 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)