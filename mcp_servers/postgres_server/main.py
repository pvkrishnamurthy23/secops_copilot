from fastmcp import FastMCP
import psycopg2
import pandas as pd
import os
import openai
from dotenv import load_dotenv
import re
from tavily import TavilyClient
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

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
        "Status has 2 values PASS and FAIL. add values to query accordingly"
        "severity has 4 values medium, critical, high, low. add values to query accordingly"
        "service_name has following values cloudformation, fms, config, support, backup, resourceexplorer2, ec2, cloudtrail, s3, account, lambda, apigateway, cloudwatch, ssm, accessanalyzer, autoscaling, iam, vpc, acm, athena, trustedadvisor"
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
    print(generated_sql)
    rows = sql_query(generated_sql)
    df = pd.DataFrame(rows)
    if df.empty:
        return "No results found for your query."
    preview = df.head(10).to_markdown(index=False)
    summary_prompt = f"Here is the SQL result table:\n\n{preview}\n\n take the content from the result and summarize it."
    summary_response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': summary_prompt}]
    )
    summary = summary_response.choices[0].message.content
    print(type(summary))
    print(summary)
    return summary





CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "website_content"
OPENAI_MODEL = "gpt-4o-mini"  # or "gpt-3.5-turbo" for cheaper responses

# Initialize embeddings and vector store
def init_vector_store():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        chunk_size=1,
        openai_api_key=api_key
    )

    vectordb = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    return vectordb

# Search vector DB and get top-k chunks
def retrieve_context(vectordb, query, k=5):
    return vectordb.similarity_search(query, k=k)

# Compose a long-form answer using OpenAI's GPT model
def generate_answer(query, context_docs):
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(openai_api_key=api_key, model=OPENAI_MODEL)

    # Concatenate the retrieved context
    context_text = "\n\n---\n\n".join(doc.page_content for doc in context_docs)

    prompt = f"""You are a helpful assistant. Use the below retrieved content to answer the user's query in detail output only in valid string format.

### Retrieved Content:
{context_text}

### User Query:
{query}

### Answer:
"""
    response = llm([HumanMessage(content=prompt)])
    return response.content


@app.tool
def PlatformWebSearchTool(query: str):
    """Search a single website for a query using Tavily and return the result."""
    query = query
    if query:
        
        try:
            vectordb = init_vector_store()
            docs = retrieve_context(vectordb, query)
            if not docs:
                return f"No relevant documents found."
            else:
                                        
                answer = generate_answer(query, docs)
                return(answer)
        except Exception as e:
            return f"Web search error: {e}"
        

@app.tool
def RemediationSuggesterTool(query: str):
    """Generate remediation steps or a custom script for a specific cloud misconfiguration using OpenAI."""
    prompt = (
        f"A user has reported the following cloud misconfiguration: '{query}'.\n"
        "Provide a step-by-step remediation plan or a custom script or terraform script to fix this issue. "
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