from typing import Literal
import uuid

from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import SystemMessage, AIMessage

from guardrails import guardrails, response_guardrails_chain
from common import llm, vector_store, OverallState

system_prompt = f"""
You are a virtual assistant for a Alaska Department of Snow (ADS) that manages snow-related public services. Your job is to assist users by answering common questions during snow events, such as road conditions, plowing schedules, school closures, and general service disruptions.
Use a clear, calm, and professional tone. Keep responses helpful, factual, and easy to understand. If users ask about something outside your knowledge or responsibilities, do not guess or make up information.

Guidelines:
- Use available tool calls or plugins to retrieve information when possible.
- Provide accurate, concise responses to routine snow-related inquiries.
- If a question is out of scope or you don’t have enough information, respond politely and clearly. For example: “I’m sorry, I don’t have that information. Please check with your local office or visit the official website for assistance.”
- Prioritize clarity, safety, and helpfulness in every interaction.
- Avoid jargon and keep language accessible to all users.
"""


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to Alaska Department of Snow (ADS)."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = f"{system_prompt}" "\n\n" f"{docs_content}"
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)

    # Response guard
    guardrails_output = response_guardrails_chain.invoke({"response": response})
    if guardrails_output.decision == "reject":
        # Inappropriate or off-topic — replace it
        return {
            "messages": [
                AIMessage(
                    "Apologies, the system generated an inappropriate or unrelated response. Please try asking again about snow-related services."
                )
            ],
            "response_guard_action": "reject",
        }

    return {"messages": [response]}


# Decided the path to take based on the guardrails output
def guardrails_condition(
    state: OverallState,
) -> Literal[END, "query_or_respond"]:
    if state.get("input_guard_action") == "end":
        return END
    elif state.get("input_guard_action") == "ok":
        return "query_or_respond"


def get_agent(thread_id=uuid.uuid4(), debug=False):
    # Build graph
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node(guardrails)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("guardrails")
    graph_builder.add_conditional_edges(
        "guardrails",
        guardrails_condition,
        {END: END, "query_or_respond": "query_or_respond"},
    )
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory, debug=debug)
    return graph.with_config({"configurable": {"thread_id": thread_id}})


def get_messages_history(graph: CompiledStateGraph, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)

    if state.values != None and "messages" in state.values:
        return state.values["messages"]

    return []


def create_new_thread():
    thread_id = uuid.uuid4()
    return str(thread_id)
