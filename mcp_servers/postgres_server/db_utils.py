import psycopg2
import os

def get_connection():
    return psycopg2.connect(os.getenv('DATABASE_URL'))

def get_finding_by_id(finding_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT * FROM findings WHERE id = %s', (finding_id,))
            return cur.fetchone() 