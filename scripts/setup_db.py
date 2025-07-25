import psycopg2
import os

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

cur.execute('''
CREATE TABLE IF NOT EXISTS findings (
    id SERIAL PRIMARY KEY,
    title TEXT,
    description TEXT,
    severity TEXT
);
''')

conn.commit()
cur.close()
conn.close() 