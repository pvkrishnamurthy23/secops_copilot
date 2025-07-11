import requests 
import json 

class MCPSession:
    def __init__(self, server_url):
        self.server_url = server_url
        self.session_id = None
        self.session = requests.Session()
    
    def initialize(self):
        """Initialize the MCP session"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "mcp-test-client", "version": "1.0.0"}
            }
        }
        
        headers = { 
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json" 
        }
        
        print("Initializing session...")
        response = self.session.post(self.server_url, json=payload, headers=headers, stream=True)
        
        if response.status_code == 200:
            # Read the stream to get the session ID
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str.strip():
                            try:
                                data = json.loads(data_str)
                                if 'result' in data:
                                    print("Session initialized successfully")
                                    # Extract session ID from response headers
                                    self.session_id = response.headers.get('mcp-session-id')
                                    print(f"Session ID: {self.session_id}")
                                    return True
                            except json.JSONDecodeError:
                                continue
        else:
            print(f"Failed to initialize session. Status: {response.status_code}")
            print(f"Response: {response.text}")
        return False
    
    def make_request(self, method, params=None, request_id=1):
        """Make a request using the established session"""
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        headers = { 
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json" 
        }
        
        # Add session ID if we have one
        if self.session_id:
            headers['X-MCP-Session-ID'] = self.session_id
        
        print(f"Making {method} request...")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print(f"Session ID: {self.session_id}")
        
        response = self.session.post(self.server_url, json=payload, headers=headers, stream=True)
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            # Read the stream
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    print(f"Raw response line: {line_str}")
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str.strip():
                            try:
                                data = json.loads(data_str)
                                return data
                            except json.JSONDecodeError:
                                continue
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
        return None

def list_tools(mcp_session): 
    print("=== Listing Available Tools ===")
    
    # List tools
    tools_result = mcp_session.make_request("tools/list", {}, 2)
    
    if tools_result and 'result' in tools_result and 'tools' in tools_result['result']:
        tools = tools_result['result']['tools']
        print(f"Found {len(tools)} tools:")
        for tool in tools:
            print(f"Tool: {tool.get('name')}")
            print(f"Description: {tool.get('description')}")
            input_schema = tool.get('inputSchema')
            if input_schema:
                print(f"Parameters: {json.dumps(input_schema, indent=2)}")
            print("-" * 40)
    else:
        print("No tools found or error occurred")
        print(f"Tools result: {tools_result}")

def call_tool(mcp_session, tool_name, arguments=None):
    """Call a specific tool with arguments"""
    print(f"=== Calling Tool: {tool_name} ===")
    
    result = mcp_session.make_request("tools/call", {
        "name": tool_name,
        "arguments": arguments or {}
    }, 3)
    
    if result and 'result' in result:
        print("Tool call successful!")
        print(f"Result: {json.dumps(result['result'], indent=2)}")
        return result['result']
    else:
        print(f"Tool call failed: {result}")
        return None

if __name__ == "__main__": 
    server_url = "http://127.0.0.1:8000/mcp"
    
    # Create and initialize session
    mcp_session = MCPSession(server_url)
    if not mcp_session.initialize():
        print("Failed to initialize session. Exiting.")
        exit(1)
    
    # List available tools
    list_tools(mcp_session)
    
    print("\n=== Testing Tool Calls ===")
    # Test calling a tool
    call_tool(mcp_session, "get_top_5_findings") 
