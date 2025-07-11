import streamlit as st
import requests
import os
import pandas as pd
import json

API_URL = os.getenv("MCP_API_URL", "http://127.0.0.1:8000/mcp/")

st.title("🛠️ MCP Tool Dashboard")

# Helper to make JSON-RPC requests
def mcp_rpc(method, params=None, id=1):
    payload = {
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params or {}
    }
    try:
        resp = requests.post(API_URL, json=payload, headers={"Content-Type": "text/event-stream"})
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            st.error(f"MCP Error: {data['error']['message']}")
            return None
        return data.get("result")
    except Exception as e:
        st.error(f"Failed to connect to MCP API: {e}")
        return None

# 1. List available tools
with st.spinner("Fetching available tools..."):
    #tools_result = mcp_rpc("tools/list")
    tools_result={ 

    "tools": [ 

      { 

        "name": "text_summarizer", 

        "description": "Summarizes input text into a concise summary.", 

        "inputSchema": { 

          "type": "object", 

          "properties": { 

            "text": { 

              "type": "string", 

              "description": "The text to be summarized." 

            }, 

            "max_length": { 

              "type": "integer", 

              "description": "Maximum length of the summary." 

            } 

          }, 

          "required": ["text"] 

        } 

      }, 

      { 

        "name": "image_classifier", 

        "description": "Classifies images into predefined categories.", 

        "inputSchema": { 

          "type": "object", 

          "properties": { 

            "image_url": { 

              "type": "string", 

              "description": "URL of the image to classify." 

            }, 

            "top_k": { 

              "type": "integer", 

              "description": "Number of top predictions to return." 

            } 

          }, 

          "required": ["image_url"] 

        } 

      }, 

      { 

        "name": "weather_forecast", 

        "description": "Provides weather forecast for a given location.", 

        "inputSchema": { 

          "type": "object", 

          "properties": { 

            "location": { 

              "type": "string", 

              "description": "Name of the city or coordinates." 

            }, 

            "days": { 

              "type": "integer", 

              "description": "Number of days to forecast." 

            } 

          }, 

          "required": ["location"] 

        } 

      } 

    ] 

  

} 

if not tools_result or "tools" not in tools_result:
    st.stop()

tools = tools_result["tools"]
tool_names = [tool["name"] for tool in tools]
tool_map = {tool["name"]: tool for tool in tools}

st.write("Available Tools:", tool_names)

# 2. Select a tool
tool_choice = st.selectbox("Choose a tool to run", tool_names)
selected_tool = tool_map[tool_choice]

# 3. Show argument fields
def get_arg_fields(tool):
    schema = tool.get("inputSchema", {})
    props = schema.get("properties", {})
    required = schema.get("required", [])
    arg_values = {}
    for arg, meta in props.items():
        label = f"{arg} ({meta.get('type', 'str')})"
        default = meta.get("default", "")
        if meta.get("type") == "integer":
            val = st.number_input(label, value=int(default) if default != "" else 0, step=1, key=arg)
        elif meta.get("type") == "number":
            val = st.number_input(label, value=float(default) if default != "" else 0.0, key=arg)
        elif meta.get("type") == "boolean":
            val = st.checkbox(label, value=bool(default), key=arg)
        else:
            val = st.text_input(label, value=str(default), key=arg)
        arg_values[arg] = val
    return arg_values

st.markdown(f"**Description:** {selected_tool.get('description', 'No description')} ")

with st.form("tool_form"):
    arg_values = get_arg_fields(selected_tool)
    submitted = st.form_submit_button("Run Tool")

# 4. Call the tool and display results
if submitted:
    with st.spinner("Calling tool..."):
        params = {"name": tool_choice, "arguments": arg_values}
        result = mcp_rpc("tools/call", params, id=2)
        if result is not None:
            # Try to display as table if possible
            content = result.get("content")
            structured = result.get("structuredContent")
            if structured:
                st.subheader("Structured Result")
                st.json(structured)
            if content:
                # If content is a list of dicts, show as table
                if isinstance(content, list) and all(isinstance(x, dict) for x in content):
                    st.subheader("Content Table")
                    st.dataframe(pd.DataFrame(content))
                else:
                    st.subheader("Content")
                    st.write(content)
        else:
            st.warning("No result returned from tool.")