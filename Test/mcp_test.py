import requests
import os
import json

API_URL = os.getenv("MCP_API_URL", "http://127.0.0.1:8000/mcp/")

def get_tools():
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    resp = requests.post(API_URL, json=payload,headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        print(f"MCP Error: {data['error']['message']}")
        return []
    return data.get("result", {}).get("tools", [])

if __name__ == "__main__":
    tools = get_tools()
    print("Available tools:")
    for tool in tools:
        print(f"- {tool['name']}")
