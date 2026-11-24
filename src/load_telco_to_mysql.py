import os
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "data/telco_churn.csv"


def clean_telco_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Fix IBM Telco quirk: 11 new customers (tenure == 0) have blank spaces in TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Convert Churn ('Yes'/'No') to binary 1/0
    df["Churn"] = (df["Churn"].astype(str).str.strip() == "Yes").astype(int)

    # Standardize column names to snake_case for SQL
    df.columns = [
        "customer_id", "gender", "senior_citizen", "partner", "dependents",
        "tenure", "phone_service", "multiple_lines", "internet_service",
        "online_security", "online_backup", "device_protection", "tech_support",
        "streaming_tv", "streaming_movies", "contract", "paperless_billing",
        "payment_method", "monthly_charges", "total_charges", "churn"
    ]
    return df


def load_to_mysql(df: pd.DataFrame) -> None:
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS telco_customers;")
    cursor.execute(
        """
        CREATE TABLE telco_customers (
            customer_id VARCHAR(32) PRIMARY KEY,
            gender VARCHAR(16),
            senior_citizen TINYINT,
            partner VARCHAR(8),
            dependents VARCHAR(8),
            tenure INT,
            phone_service VARCHAR(8),
            multiple_lines VARCHAR(32),
            internet_service VARCHAR(32),
            online_security VARCHAR(32),
            online_backup VARCHAR(32),
            device_protection VARCHAR(32),
            tech_support VARCHAR(32),
            streaming_tv VARCHAR(32),
            streaming_movies VARCHAR(32),
            contract VARCHAR(32),
            paperless_billing VARCHAR(8),
            payment_method VARCHAR(64),
            monthly_charges DECIMAL(10, 2),
            total_charges DECIMAL(10, 2),
            churn TINYINT
        );
        """
    )

    cols = ", ".join(df.columns)
    placeholders = ", ".join(["%s"] * len(df.columns))
    insert_sql = f"INSERT INTO telco_customers ({cols}) VALUES ({placeholders})"

    # Convert numpy types to native Python types for mysql-connector
    records = [
        tuple(v.item() if hasattr(v, "item") else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    cursor.executemany(insert_sql, records)
    conn.commit()

    cursor.execute("SELECT COUNT(*), ROUND(AVG(churn) * 100, 2) FROM telco_customers;")
    total_rows, churn_pct = cursor.fetchone()
    print(f"Loaded {total_rows:,} rows into MySQL table 'telco_customers' (Churn rate: {churn_pct}%).")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    telco_df = clean_telco_data(CSV_PATH)
    load_to_mysql(telco_df)