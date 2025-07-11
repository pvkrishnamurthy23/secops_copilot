import os
import psycopg2
import csv
from dotenv import load_dotenv

load_dotenv()

def ingest_prowler_report(report_path):
    # Parse Prowler report CSV, create table, and store in Postgres
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('DATABASE_URL not set in environment.')
        return
    try:
        conn = psycopg2.connect(db_url)
        with open(report_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            headers = next(reader)
            # Clean column names (remove spaces, lower case)
            columns = [h.strip().replace(' ', '_').lower() for h in headers]
            # Drop table if exists
            with conn.cursor() as cur:
                cur.execute('DROP TABLE IF EXISTS findings;')
                # Create table with all columns as TEXT
                col_defs = ', '.join([f'{col} TEXT' for col in columns])
                cur.execute(f'CREATE TABLE findings ({col_defs});')
                # Insert data rows
                for row in reader:
                    placeholders = ', '.join(['%s'] * len(row))
                    cur.execute(
                        f"INSERT INTO findings ({', '.join(columns)}) VALUES ({placeholders});",
                        row
                    )
                conn.commit()
        print('Table created and data ingested successfully.')
    except Exception as e:
        print(f'Error during ingestion: {e}')
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    import sys
    ingest_prowler_report(r"C:\secops_copilot\File\prowler-output.csv") 