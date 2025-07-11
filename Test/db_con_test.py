import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DB_URL = os.getenv('DATABASE_URL')

def get_connection():
    """Establish a connection to the PostgreSQL database using env variable."""
    try:
        conn = psycopg2.connect(DB_URL)
        print("Connection successful!")
        return conn
    except Exception as e:
        print(f"Connection failed: {e}")
        return None

def create_table():
    """Create a sample table if it doesn't exist."""
    conn = get_connection()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS test_table (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    age INTEGER
                );
            ''')
            conn.commit()
            print("Table created or already exists.")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        conn.close()

def insert_data(name, age):
    """Insert a row into the test_table."""
    conn = get_connection()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute('INSERT INTO test_table (name, age) VALUES (%s, %s);', (name, age))
            conn.commit()
            print(f"Inserted ({name}, {age}) into test_table.")
    except Exception as e:
        print(f"Error inserting data: {e}")
    finally:
        conn.close()

def test_db_connection():
    """Test the database connection."""
    conn = get_connection()
    if conn:
        conn.close()

def get_findings_schema():
    """Print the schema (column names and types) of the findings table."""
    conn = get_connection()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'findings';")
            schema = cur.fetchall()
            print("findings table schema:")
            for col, dtype in schema:
                print(f"  {col}: {dtype}")
    except Exception as e:
        print(f"Error fetching schema: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    test_db_connection()
    create_table()
    insert_data("Alice", 30)
    insert_data("Bob", 25)
