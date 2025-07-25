from fastmcp import FastMCP
import psycopg2
import os


app = FastMCP()

def get_connection():
    return psycopg2.connect(os.getenv('DATABASE_URL'))

@app.tool
def get_finding(id: int):
    """Get a single finding by id from the findings table."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT * FROM findings WHERE id = %s', (id,))
            row = cur.fetchone()
            if row is None:
                return None
            # Get column names
            if cur.description is None:
                return None
            colnames = [desc[0] for desc in cur.description]
            return dict(zip(colnames, row))

@app.tool
def get_all_findings():
    """Get all findings from the findings table."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT * FROM findings')
            rows = cur.fetchall()
            if cur.description is None:
                return []
            colnames = [desc[0] for desc in cur.description]
            return [dict(zip(colnames, row)) for row in rows]

@app.tool
def get_top_5_findings():
    """Get the top 5 findings from the findings table."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT * FROM findings LIMIT 5')
            rows = cur.fetchall()
            if cur.description is None:
                return []
            colnames = [desc[0] for desc in cur.description]
            return [dict(zip(colnames, row)) for row in rows]

@app.tool
def get_top_10_findings():
    """Get the top 10 findings from the findings table."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT * FROM findings LIMIT 10')
            rows = cur.fetchall()
            if cur.description is None:
                return []
            colnames = [desc[0] for desc in cur.description]
            return [dict(zip(colnames, row)) for row in rows]

@app.tool
def list_tables():
    """List all table names in the connected database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            rows = cur.fetchall()
            print(rows)
            print(type(rows))
            return [row[0] for row in rows]

@app.tool
def sql_query(query: str):
    """Execute a SQL query and return the results as a list of dicts."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            if cur.description is None:
                return []
            colnames = [desc[0] for desc in cur.description]
            return [dict(zip(colnames, row)) for row in rows]

@app.tool
def get_schema():
    """Return a list of all tables and their columns (name and data type) for the connected database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            tables = [row[0] for row in cur.fetchall()]
            schema = []
            for table in tables:
                cur.execute(
                    """
                    SELECT column_name, data_type FROM information_schema.columns
                    WHERE table_name = %s
                    """,
                    (table,)
                )
                columns = [
                    {"column_name": col[0], "data_type": col[1]}
                    for col in cur.fetchall()
                ]
                schema.append({"table": table, "columns": columns})
            return schema

if __name__ == '__main__':
    print("Starting FastMCP server on http://127.0.0.1:8000/mcp")
    app.run(transport='streamable-http', host='127.0.0.1', port=8000)