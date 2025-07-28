import logging
import os
import asyncio
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- LangChain Q&A Agent ----------
def create_llm(config: Dict[str, Any]) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.get("openai_model", "gpt-4o-mini"),
        temperature=config.get("llm_temperature", 0.7),
        max_tokens=config.get("llm_max_tokens", None),
        timeout=config.get("llm_timeout", None),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

def create_system_prompt() -> str:
    """Create the system prompt with knowledge sources"""
    return (
        """You are a specialized cybersecurity assistant operating within a Managed Control Plane (MCP) system.\n\n"
        "You have access to the following intelligent tools:\n"
        "1. **FindingsInterpreterTool**: Accepts user input in natural language (e.g., 'What issues exist in my AWS environment?') and converts it into structured query results about findings or misconfigurations detected in the environment.\n"
        "2. **RemediationSuggesterTool**: Given a specific finding or misconfiguration ID or description (e.g., 'Public S3 bucket', or ID: `FND-023`), it returns actionable remediation guidance.\n"
        "3. **PlatformWebSearchTool**: Performs a web search to answer general platform-related questions (e.g., 'How do I integrate Azure AD with MCP?').\n\n"
        "INSTRUCTIONS:\n"
        "1. Start by analyzing the user's question carefully.\n"
        "2. If the input is related to *asking about security issues, misconfigurations, or findings in the environment*, use **FindingsInterpreterTool** to extract the structured query.\n"
        "3. If the input is about *remediation steps for a specific misconfiguration or finding*, use **RemediationSuggesterTool**.\n"
        "4. If the input is a *general platform-related or integration question*, use **PlatformWebSearchTool**.\n"
        "5. If the input is ambiguous or missing important context (e.g., 'How do I fix this?' without saying what “this” is), ask a clear follow-up question to get required details.\n"
        "6. NEVER assume missing information. Always clarify before proceeding.\n"
        "7. Only respond with answers when confident. Otherwise, request more input.\n\n"
        "TOOL USAGE STRATEGY:\n"
        "- **First**, analyze intent: Is this about findings, remediation, or platform help?\n"
        "- **Second**, use the appropriate tool to process the request.\n"
        "- **Third**, if the response from the tool is not enough, ask the user for more details.\n\n"
        "RULES:\n"
        "- Be clear, precise, and helpful.\n"
        "- Never guess. Always clarify vague inputs.\n"
        "- Always use tools in the reasoning loop before giving a final answer.\n"
        "- Cite sources or reasoning where possible.\n\n"
        "TOOLS YOU CAN USE:\n"
        "{{tools}}\n\n"
        "Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\n"
        "Valid 'action' values: 'Final Answer' or {{tool_names}}\n\n"
        "Provide only ONE action per $JSON_BLOB, as shown:\n"
        {{{{  
  "action": "$TOOL_NAME",  
  "action_input": {{  
    "query": "value1",  
    
  }}  
}}}}
        "Follow this format:\n\n"
        "Question: input question to answer\n"
        "Thought: consider previous and subsequent steps\n"
        "Action:\n```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action: 
```
{{{{
  "action": "Final Answer",
  "action_input": "response"
}}}}
```
Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate and ask for clarification if something is not clear. Format is Action:```$JSON_BLOB```then Observation
"""
    )

class DomainQAAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            raise ValueError("Configuration is required")
        self.config = config
        self.mcp_server_url = config.get("mcp_server_url", "http://127.0.0.1:8000/mcp")
        self.llm = create_llm(config)
        self.chat_history: List[BaseMessage] = []
        self.mcp_client = None
        self.tools = []
        self.agent_executor = None

    async def _initialize_mcp_client(self):
        if self.mcp_client is None:
            connection = StreamableHttpConnection(url=self.mcp_server_url, transport="streamable_http")
            self.mcp_client = MultiServerMCPClient(connections={"qa_agent_server": connection})
            self.tools = await self.mcp_client.get_tools()
            self.agent_executor = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        system_message = create_system_prompt()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                (
                    "human",
                    "{input}\n\n{agent_scratchpad}(reminder to respond in a JSON blob no matter what)...",
                ),
            ]
        ).partial(
            tools="\n".join([f"- {tool.name}" for tool in self.tools]),
            tool_names=", ".join([tool.name for tool in self.tools])
        )

        agent = create_structured_chat_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=15,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

    async def achat(self, user_input: str) -> str:
        await self._initialize_mcp_client()
        agent_input = {
            "input": user_input,
            "chat_history": self.chat_history[-5:],
        }
        response = await self.agent_executor.ainvoke(agent_input)
        answer = response.get("output", "I couldn't process your request.")
        self.chat_history.extend([
            HumanMessage(content=user_input), AIMessage(content=answer)
        ])
        return answer

    async def close(self):
        if self.mcp_client:
            await self.mcp_client.close()

# ---------- Streamlit Chat UI ----------
st.set_page_config(page_title="SecOps Co-Pilot", page_icon="🛡️")
st.title("Calibo Accelerate SecOPS Assistant")

if "agent" not in st.session_state:
    config = {
        "openai_model": "gpt-4o-mini",
        "mcp_server_url": "http://127.0.0.1:8000/mcp"
    }
    st.session_state.agent = DomainQAAgent(config=config)
    st.session_state.history = []

user_input = st.chat_input("Ask your question about findings, remediation, or platform...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = asyncio.run(st.session_state.agent.achat(user_input))
        st.markdown(response)

    # Update session history
    st.session_state.history.append(HumanMessage(content=user_input))
    st.session_state.history.append(AIMessage(content=response))
    st.session_state.agent.chat_history = st.session_state.history[-5:]
