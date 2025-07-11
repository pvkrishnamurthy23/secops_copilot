from fastmcp import MCPServer
from .rag_utils import query_rag

app = MCPServer()

@app.method()
def search_idp(query: str):
    return query_rag(query)

if __name__ == '__main__':
    app.run() 