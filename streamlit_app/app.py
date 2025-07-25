import streamlit as st
import os
import pandas as pd
import openai
import re
import asyncio
from fastmcp import Client
from dotenv import load_dotenv


load_dotenv()

API_URL = os.getenv("MCP_API_URL", "http://127.0.0.1:8000/mcp/")
openai.api_key = os.getenv('OPENAI_API_KEY')

# Helper: FastMCP client async fetch for SQL query
def fetch_sql_sync(query):
    async def fetch():
        async with Client(API_URL) as client:
            result = await client.call_tool("sql_query", {"query": query})
            if result.structured_content:
                return result.structured_content
            elif result.content:
                return result.content
            return []
    return asyncio.run(fetch())

# Helper: Fetch schema details from MCP server
def fetch_schema_sync():
    async def fetch():
        async with Client(API_URL) as client:
            # Try to call a tool named 'get_schema' or similar
            try:
                result = await client.call_tool("get_schema", {})
                if result.structured_content:
                    return result.structured_content
                elif result.content:
                    return result.content
            except Exception:
                pass
            # Fallback: try to infer schema from all tables
            try:
                tables_result = await client.call_tool("list_tables", {})
                tables = tables_result.structured_content or tables_result.content or []
                schema_info = []
                for table in tables:
                    if isinstance(table, dict):
                        table_name = next(iter(table.values()), None)
                    else:
                        table_name = table
                    if not table_name:
                        continue
                    # Try to get columns for each table
                    try:
                        col_result = await client.call_tool("sql_query", {"query": f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"})
                        columns = col_result.structured_content or col_result.content or []
                        schema_info.append({"table": table_name, "columns": columns})
                    except Exception:
                        continue
                return schema_info
            except Exception:
                return []
    return asyncio.run(fetch())

def ask_openai(prompt, model='gpt-4o-mini'):
    response = openai.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content

st.title("SecOPS Co-Pilot")

# Fetch schema details once and cache in session state
if 'schema_details' not in st.session_state:
    with st.spinner("Fetching schema details from MCP server..."):
        st.session_state['schema_details'] = fetch_schema_sync()

schema_details = st.session_state['schema_details']

# Format schema details for OpenAI context
schema_context = ""
if schema_details:
    if isinstance(schema_details, list):
        for table in schema_details:
            if isinstance(table, dict) and 'table' in table and 'columns' in table:
                schema_context += f"Table: {table['table']}\n"
                for col in table['columns']:
                    if isinstance(col, dict):
                        schema_context += f"  - {col.get('column_name', '')}: {col.get('data_type', '')}\n"
                    elif isinstance(col, (list, tuple)) and len(col) == 2:
                        schema_context += f"  - {col[0]}: {col[1]}\n"
                schema_context += "\n"
            elif isinstance(table, str):
                schema_context += f"Table: {table}\n"
    elif isinstance(schema_details, str):
        schema_context = schema_details

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Chat input
user_input = st.chat_input("Ask a data question in natural language...")

if user_input:
    # Add user message to chat history
    st.session_state['messages'].append({"role": "user", "content": user_input})

    # 1. Generate SQL from user question, providing schema context
    system_prompt = (
        "You are a helpful data analyst. "
        "Table name is 'findings'. "
        "The database consists of plowler reports on AWS account and its findings"
        "restrict the number of results to 10 for all queries"
        "column names are  assessment_start_time,finding_unique_id,provider, check_id, check_title, check_type, status, status_extended, service_name, subservice_name, severity, resource_type, resource_details, resource_tags, description, risk, related_url, remediation_recommendation_text, remediation_recommendation_url, remediation_recommendation_code_nativeiac, remediation_recommendation_code_terraform, remediation_recommendation_code_cli, remediation_recommendation_code_other, compliance, categories, depends_on, related_to, notes, profile, account_id, account_name, account_email, account_arn, account_org, account_tags, region, resource_id, resource_arn"
        "Given the above database schema, generate a valid PostgreSQL SQL query to answer the user's question. "
        "Return only SQL. and nothing else. Don't explain.\n\n"
      #  f"Database schema:\n{schema_context}\n"
    )
    full_prompt = f"{system_prompt}\nUser: {user_input}"
    sql_response = ask_openai(full_prompt)

    # Try to extract SQL code from the response
    sql_match = re.search(r"```sql(.*?)```", sql_response or "", re.DOTALL | re.IGNORECASE)
    if sql_match:
        generated_sql = sql_match.group(1).strip()
    else:
        generated_sql = (sql_response or "").strip('`').strip()

    # 2. Execute SQL via MCP
    try:
        print(generated_sql)
        #print(st.session_state['schema_details'])
        rows = fetch_sql_sync(generated_sql)
        df = pd.DataFrame(rows)
        # Show SQL and table in chat
        st.session_state['messages'].append({"role": "assistant", "content": f"SQL Query:\n```sql\n{generated_sql}\n```"})
        if not df.empty:
            # 3. Summarize result with OpenAI
            preview = df.head(10).to_markdown(index=False)
            summary_prompt = f"Here is the SQL result table:\n\n{preview}\n\nSummarize the result for the user in a concise, friendly way."
            summary = ask_openai(summary_prompt)
            st.session_state['messages'].append({"role": "assistant", "content": summary})
        else:
            st.session_state['messages'].append({"role": "assistant", "content": "No results found for your query."})
    except Exception as e:
        st.session_state['messages'].append({"role": "assistant", "content": f"Error executing SQL: {e}"})

# Display chat history
for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        st.chat_message('user').write(msg['content'])
    else:
        st.chat_message('assistant').write(msg['content'])