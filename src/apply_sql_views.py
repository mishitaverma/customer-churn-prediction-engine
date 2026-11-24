import os
import mysql.connector
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def apply_views() -> None:
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )
    cursor = conn.cursor()

    with open("sql/analytics_views.sql", "r") as f:
        sql_script = f.read()

    for statement in sql_script.split(";"):
        stmt = statement.strip()
        if stmt:
            cursor.execute(stmt)

    conn.commit()
    print("Created views: v_churn_by_tenure, v_churn_by_payment, v_churn_by_segment\n")

    for view in ["v_churn_by_tenure", "v_churn_by_payment"]:
        cursor.execute(f"SELECT * FROM {view};")
        df = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])
        print(f"=== {view} ===")
        print(df.to_string(index=False), "\n")

    cursor.execute("SELECT * FROM v_churn_by_segment LIMIT 5;")
    seg_df = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])
    print("=== Top 5 High-Risk Segments (v_churn_by_segment) ===")
    print(seg_df.to_string(index=False))

    cursor.close()
    conn.close()


if __name__ == "__main__":
    apply_views()