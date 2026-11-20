import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("1. Extracting data from MySQL...")
try:
    connection = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )
    
    # Read directly from SQL into a Pandas DataFrame
    query = "SELECT plan, support_tickets, days_inactive, churned FROM ml_customers"
    df = pd.read_sql(query, connection)
    print(f"Successfully loaded {len(df)} records.\n")

except Exception as e:
    print(f"Database Error: {e}")
    exit()
finally:
    if 'connection' in locals() and connection.is_connected():
        connection.close()

print("2. Preprocessing data...")
# One-Hot Encode the 'plan' column (Basic, Pro, Enterprise)
df = pd.get_dummies(df, columns=['plan'], drop_first=True)

# Separate features (X) and target (y)
X = df.drop('churned', axis=1)
y = df['churned']

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("4. Evaluating Model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
import joblib

# Save the trained model and feature column names
joblib.dump(model, 'churn_model.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')
print("Model and feature columns saved successfully!")