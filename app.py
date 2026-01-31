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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- PAGE CONFIG ---
st.set_page_config(page_title="ML Classification Cockpit", page_icon="🛡️", layout="wide")

# --- CUSTOM LOGGING SYSTEM (The v1.3 Feature) ---
if 'system_logs' not in st.session_state:
    st.session_state.system_logs = []


def st_log(message, level="INFO"):
    """Adds a timestamped message to the session state log."""
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{level}] {message}"
    st.session_state.system_logs.append(entry)


# --- STEP 1: LOAD & PREPROCESS DATA ---
@st.cache_data
def load_data():
    try:
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, data.target_names, data.feature_names
    except Exception as e:
        st_log(f"Failed to load dataset: {e}", "CRITICAL")
        return None, None, None


def preprocess_data(df):
    # INPUT VALIDATION (Error Handling)
    if df is None or df.empty:
        raise ValueError("Dataset is empty or invalid.")

    X = df.drop('target', axis=1)
    y = df['target']

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# --- UI HEADER ---
st.title("🛡️ MCert Classification Cockpit (Robust)")
st.markdown("Comparing **Logistic**, **Tree**, **SVM**, and **KNN** with **Production-Grade Error Handling**.")
st.markdown("---")

# Load Data
df, target_names, feature_names = load_data()

if df is not None:
    st_log("Dataset loaded successfully. Ready for operations.", "INFO")
else:
    st.error("Critical Error: Could not load data.")
    st.stop()

# --- SIDEBAR: DATA EXPLORER ---
with st.sidebar:
    st.header("🗂️ Data Control")
    if st.checkbox("Show Raw Data Sample"):
        st.dataframe(df.head(10))

    st.info(f"Total Samples: {df.shape[0]}")

    st.divider()
    st.header("📝 System Logs")
    # Display the last 5 logs in the sidebar
    log_text = "\n".join(st.session_state.system_logs[-10:])
    st.text_area("Console Output", log_text, height=200, disabled=True)

# --- MAIN COCKPIT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Model Configuration")
    model_choice = st.radio(
        "Select Algorithm:",
        ["Logistic Regression", "Decision Tree", "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)"]
    )

    st.markdown("---")
    st.write("**Hyperparameter Tuning**")

    if model_choice == "Decision Tree":
        max_depth = st.slider("Max Tree Depth", 1, 10, 3)
    elif model_choice == "K-Nearest Neighbors (KNN)":
        k_neighbors = st.slider("Number of Neighbors (K)", 1, 21, 5, step=2)

    run_btn = st.button("🚀 Train & Evaluate", type="primary")

with col2:
    st.subheader("📊 Performance Metrics")

    if run_btn:
        st_log(f"Starting training process for {model_choice}...", "INFO")

        # ERROR HANDLING WRAPPER (The v1.3 Feature)
        try:
            # 1. Prepare Data
            with st.status("🛠️ Preprocessing & Training...", expanded=True) as status:
                X_train, X_test, y_train, y_test = preprocess_data(df)
                st_log("Data preprocessing and scaling complete.", "DEBUG")

                # 2. Initialize Model
                if model_choice == "Logistic Regression":
                    model = LogisticRegression()
                elif model_choice == "Decision Tree":
                    model = DecisionTreeClassifier(max_depth=max_depth if 'max_depth' in locals() else 3)
                elif model_choice == "Support Vector Machine (SVM)":
                    model = SVC()
                else:
                    model = KNeighborsClassifier(n_neighbors=k_neighbors if 'k_neighbors' in locals() else 5)

                # 3. Train Model
                model.fit(X_train, y_train)
                st_log("Model training converged successfully.", "SUCCESS")
                status.update(label="✅ Training Complete", state="complete", expanded=False)

            # 4. Evaluation
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.2%}")
            m2.metric("Precision", f"{prec:.2%}")
            m3.metric("Recall", f"{rec:.2%}")
            m4.metric("F1 Score", f"{f1:.2%}")

            # 5. Visualization
            st.markdown("---")
            if model_choice == "Decision Tree":
                st.subheader("🌳 Decision Tree Logic Flow")
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(model, filled=True, feature_names=feature_names, class_names=target_names, fontsize=10)
                st.pyplot(fig)
            else:
                st.subheader("📉 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)

        except ValueError as ve:
            # Catch Data Errors
            st.error(f"Data Validation Error: {ve}")
            st_log(f"Validation Error: {ve}", "ERROR")

        except Exception as e:
            # Catch Runtime/Model Errors
            st.error(f"Unexpected System Crash: {e}")
            st_log(f"CRITICAL FAILURE: {e}", "CRITICAL")