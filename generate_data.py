import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from faker import Faker
import random
import mysql.connector

fake = Faker()

def generate_advanced_data(num_records=5000):
    print(f"Generating {num_records} customer records...")
    data = []
    for _ in range(num_records):
        # AI Features: How many complaints? How long since they logged in?
        tickets = random.randint(0, 8)
        days_inactive = random.randint(1, 45)
        
        # Rule: People who don't log in and complain a lot are more likely to cancel (churn)
        if tickets > 4 and days_inactive > 20:
            will_cancel = True
        else:
            will_cancel = random.choice([True, False, False, False, False]) # 20% chance otherwise
            
        data.append((
            fake.name(),
            random.choice(['Basic', 'Pro', 'Enterprise']),
            tickets,
            days_inactive,
            will_cancel
        ))
    return data

records = generate_advanced_data()

try:
    connection = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )
    cursor = connection.cursor()

    # Build a bigger, better compartment for Machine Learning
    cursor.execute("DROP TABLE IF EXISTS ml_customers")
    create_table_query = """
    CREATE TABLE ml_customers (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255),
        plan VARCHAR(50),
        support_tickets INT,
        days_inactive INT,
        churned BOOLEAN
    )
    """
    cursor.execute(create_table_query)

    # Use 'executemany' to insert all 5,000 at once super fast
    print("Saving to database...")
    insert_query = "INSERT INTO ml_customers (name, plan, support_tickets, days_inactive, churned) VALUES (%s, %s, %s, %s, %s)"
    cursor.executemany(insert_query, records)
    
    connection.commit()
    print("Success! 5,000 records saved for Machine Learning.")

except Exception as e:
    print(f"Error: {e}")
finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()