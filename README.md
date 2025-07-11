# Secops Copilot

A security operations chatbot for your internal developer platform, featuring:
- Streamlit chat UI
- FastMCP servers (Postgres for Prowler data, RAG for IDP text)
- OpenAI LLM integration
- Prowler data ingestion

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in your secrets.
3. Run the MCP servers and Streamlit app as described below.

## Structure
- `streamlit_app/`: Chat UI
- `mcp_servers/`: FastMCP servers (Postgres, RAG)
- `ingestion/`: Prowler data ingestion
- `llm_integration/`: OpenAI client
- `config/`: Configuration
- `scripts/`: Setup scripts 