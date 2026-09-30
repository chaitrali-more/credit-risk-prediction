from fastapi import FastAPI
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sqlite3

app = FastAPI()

# ---------- Create Database ----------
conn = sqlite3.connect("credit_risk.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    income INTEGER,
    loan INTEGER,
    score INTEGER,
    result TEXT
)
""")

conn.commit()

# ---------- Sample Dataset ----------
data = {
    "Age": [25, 45, 35, 50, 23, 40, 60, 30],
    "Income": [30000, 60000, 50000, 80000, 25000, 70000, 90000, 40000],
    "LoanAmount": [10000, 20000, 15000, 25000, 12000, 22000, 30000, 18000],
    "CreditScore": [650, 700, 680, 720, 640, 710, 750, 690],
    "Risk": [0, 0, 0, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

X = df[["Age", "Income", "LoanAmount", "CreditScore"]]
y = df["Risk"]

model = LogisticRegression()
model.fit(X, y)

# ---------- Home ----------
@app.get("/")
def home():
    return {"message": "Credit Risk Prediction with Database is running"}

# ---------- Predict and Save ----------
@app.get("/predict")
def predict(age: int, income: int, loan: int, score: int):

    prediction = model.predict([[age, income, loan, score]])

    if prediction[0] == 0:
        result = "Low Credit Risk"
    else:
        result = "High Credit Risk"

    # Save to database
    cursor.execute(
        "INSERT INTO predictions (age, income, loan, score, result) VALUES (?, ?, ?, ?, ?)",
        (age, income, loan, score, result)
    )

    conn.commit()

    return {"Risk": result}

# ---------- View Records ----------
@app.get("/records")
def get_records():

    cursor.execute("SELECT * FROM predictions")
    rows = cursor.fetchall()

    return {"Records": rows}