import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI, Request 
from mcp_servers.postgres_server.mcpServer import list_tables, run_query 


app = FastAPI() 

@app.get("/mcp/table://list") 
def list_tables_endpoint(): 
    return list_tables() 

@app.post("/mcp/sql://query") 
async def run_query_endpoint(request: Request): 
    body = await request.json() 
    return run_query(query=body["query"]) 