from fastmcp import FastMCP
import psycopg2
import pandas as pd
import logging
import os
import openai
from dotenv import load_dotenv
import re
from tavily import TavilyClient

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

app = FastMCP()

# Tavily client initialization
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def get_connection():
    return psycopg2.connect(os.getenv('DATABASE_URL'))

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
def FindingsInterpreterTool(query: str):
    """Convert a natural language question to a PostgreSQL SQL query for the findings table, execute it, and summarize the result."""
    system_prompt = (
        "You are a helpful data analyst. "
        "Table name is 'findings'. "
        "The database consists of plowler reports on AWS account and its findings"
        "restrict the number of results to 10 for all queries"
        "column names are  assessment_start_time,finding_unique_id,provider, check_id, check_title, check_type, status, status_extended, service_name, subservice_name, severity, resource_type, resource_details, resource_tags, description, risk, related_url, remediation_recommendation_text, remediation_recommendation_url, remediation_recommendation_code_nativeiac, remediation_recommendation_code_terraform, remediation_recommendation_code_cli, remediation_recommendation_code_other, compliance, categories, depends_on, related_to, notes, profile, account_id, account_name, account_email, account_arn, account_org, account_tags, region, resource_id, resource_arn"
        "Given the above database schema, generate a valid PostgreSQL SQL query to answer the user's question. "
        "Return only SQL. and nothing else. Don't explain.\n\n"
    )
    full_prompt = f"{system_prompt}\nUser: {query}"
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': full_prompt}]
    )
    sql_response = response.choices[0].message.content
    # Extract SQL code block if present
    sql_match = re.search(r"```sql(.*?)```", sql_response or "", re.DOTALL | re.IGNORECASE)
    if sql_match:
        generated_sql = sql_match.group(1).strip()
    else:
        generated_sql = (sql_response or "").strip('`').strip()

    rows = sql_query(generated_sql)
    df = pd.DataFrame(rows)
    if df.empty:
        return "No results found for your query."
    preview = df.head(10).to_markdown(index=False)
    summary_prompt = f"Here is the SQL result table:\n\n{preview}\n\nSummarize the result for the user in a concise, friendly way."
    summary_response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': summary_prompt}]
    )
    summary = summary_response.choices[0].message.content
    return summary

@app.tool
def PlatformWebSearchTool(query: str, site: str = "https://help.calibo.com/lazsa/content/home.htm"):
    """Search a single website for a query using Tavily and return the result."""
    try:
        search_results = tavily_client.search(
            query=query,
            max_results=5,
            search_depth="basic",
            include_domains=[site],
        )
        results = search_results.get("results", [])
        if not results:
            return "No results found."
        # Format the first result for brevity
        first = results[0]
        print(first)
        return f"Title: {first.get('title', 'No title')}\nURL: {first.get('url', 'No URL')}\nContent: {first.get('content', 'No content available')}"
    except Exception as e:
        return f"Web search error: {e}"

@app.tool
def RemediationSuggesterTool(query: str):
    """Generate remediation steps or a custom script for a specific cloud misconfiguration using OpenAI."""
    prompt = (
        f"A user has reported the following cloud misconfiguration: '{query}'.\n"
        "Provide a step-by-step remediation plan or a custom script to fix this issue. "
        "If a script is appropriate, provide it in a code block. Be concise, accurate, and actionable."
    )
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    print("Starting FastMCP server on http://127.0.0.1:8000/mcp")
    app.run(transport='streamable-http', host='127.0.0.1', port=8000)