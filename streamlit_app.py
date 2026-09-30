import streamlit as st
import sqlite3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -----------------------------
# TITLE
# -----------------------------

st.title("Credit Risk Prediction System")
st.subheader("Machine Learning with MLOps Deployment")

# -----------------------------
# LOAD DATASET
# -----------------------------

DATA_FILE = "credit_risk_dataset_200000_big_range.csv"

data = pd.read_csv(DATA_FILE)

# Convert Risk text to numbers
data["Risk"] = data["Risk"].map({
    "Low Credit Risk": 0,
    "High Credit Risk": 1
})

st.success(f"Total Records in Dataset: {len(data)}")

st.header("Dataset Preview (First 100 Rows)")
st.dataframe(data.head(100))

# -----------------------------
# TRAIN MODEL
# -----------------------------

X = data[["Age", "Income", "Loan", "CreditScore"]]
y = data["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.metric(
    label="Model Accuracy",
    value=f"{round(accuracy*100, 2)} %"
)

# -----------------------------
# DATABASE
# -----------------------------

conn = sqlite3.connect("credit_risk.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    income REAL,
    loan REAL,
    score INTEGER,
    result TEXT
)
""")

# -----------------------------
# USER INPUT
# -----------------------------

st.sidebar.header("Enter Customer Details")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=75,
    value=30
)

income = st.sidebar.number_input(
    "Income (₹)",
    min_value=0.0,
    value=50000.0
)

loan = st.sidebar.number_input(
    "Loan Amount (₹)",
    min_value=0.0,
    value=20000.0
)

score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

# -----------------------------
# PREDICTION (BALANCED LOGIC)
# -----------------------------

if st.sidebar.button("Predict Risk"):

    prediction = model.predict([[age, income, loan, score]])

    probability = model.predict_proba(
        [[age, income, loan, score]]
    )

    risk_prob = probability[0][1]

    # -------- REALISTIC RULES --------

    if score < 550:
        result = "High Credit Risk"

    elif loan > income * 3:
        result = "High Credit Risk"

    elif prediction[0] == 1:
        result = "High Credit Risk"

    else:
        result = "Low Credit Risk"

    # -------- DISPLAY --------

    if result == "High Credit Risk":
        st.error(result)
    else:
        st.success(result)

    st.write(
        f"Risk Probability: {round(risk_prob*100, 2)} %"
    )

    # -------- SAVE --------

    cursor.execute(
        """
        INSERT INTO predictions
        (age, income, loan, score, result)
        VALUES (?, ?, ?, ?, ?)
        """,
        (age, income, loan, score, result)
    )

    conn.commit()

# -----------------------------
# SHOW RECORDS
# -----------------------------

st.header("Saved Predictions")

records = pd.read_sql_query(
    "SELECT * FROM predictions",
    conn
)

st.dataframe(records)

# -----------------------------
# VISUALIZATION
# -----------------------------

st.header("Risk Distribution")

if not records.empty:

    risk_counts = records["result"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        risk_counts.index,
        risk_counts.values
    )

    ax.set_title("Credit Risk Distribution")

    ax.set_xlabel("Risk Type")

    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)

conn.close()