import streamlit as st
import requests
import os
import pandas as pd
import json
import asyncio
from fastmcp import Client

MCP_SERVER_URL = os.getenv("MCP_API_URL", "http://127.0.0.1:8000/mcp/")

st.title("🛠️ MCP Tool Dashboard")


async def get_tools():
    # Connect to the MCP server
    async with Client(MCP_SERVER_URL) as client:
        # List available tools
        tools = await client.list_tools()
        return tools


async def call_tool(tool_name, arguments):
    # Connect to the MCP server and call the tool
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(tool_name, arguments)
        return result


# 1. List available tools
with st.spinner("Fetching available tools..."):
    # Run the async function to get tools
    tools = asyncio.run(get_tools())

if not tools:
    st.error("No tools available or failed to connect to MCP server")
    st.stop()

tool_names = [tool.name for tool in tools]
tool_map = {tool.name: tool for tool in tools}

st.write("Available Tools:", tool_names)

# 2. Select a tool
tool_choice = st.selectbox("Choose a tool to run", tool_names)
selected_tool = tool_map[tool_choice]

# 3. Show argument fields
def get_arg_fields(tool):
    schema = tool.inputSchema
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

st.markdown(f"**Description:** {selected_tool.description or 'No description'} ")

with st.form("tool_form"):
    arg_values = get_arg_fields(selected_tool)
    submitted = st.form_submit_button("Run Tool")

# 4. Call the tool and display results
if submitted:
    with st.spinner("Calling tool..."):
        try:
            result = asyncio.run(call_tool(tool_choice, arg_values))
            
            if result:
                # Display structured content if available
                if result.structured_content:
                    st.subheader("Structured Result")
                    st.json(result.structured_content)
                
                # Display content if available
                if result.content:
                    # If content is a list of dicts, show as table
                    if isinstance(result.content, list) and all(isinstance(x, dict) for x in result.content):
                        st.subheader("Content Table")
                        st.dataframe(pd.DataFrame(result.content))
                    else:
                        st.subheader("Content")
                        st.write(result.content)
                
                # If no structured content or content, show the result object
                if not result.structured_content and not result.content:
                    st.subheader("Result")
                    st.write(result)
            else:
                st.warning("No result returned from tool.")
        except Exception as e:
            st.error(f"Error calling tool: {str(e)}")