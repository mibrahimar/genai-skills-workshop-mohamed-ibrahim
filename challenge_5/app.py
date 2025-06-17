import uuid
import streamlit as st
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage

from agent import get_agent

graph_cache: dict[str, CompiledStateGraph] = {}


@st.cache_resource
def load_graph(thread_id):
    if thread_id not in graph_cache:
        print(f"creating new graph thread_id={thread_id}")
        graph_cache[thread_id] = get_agent(thread_id, debug=True)

    return graph_cache[thread_id]


def message_chunk_to_str(input):
    if input and input is not None:
        for chunk in input:
            if (
                chunk[0].content
                and not isinstance(chunk[0], HumanMessage)
                and chunk[1]["langgraph_node"]
                in ("query_or_respond", "generate", "guardrails")
            ):
                yield chunk[0].content


def main():

    # Create session id on load
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4()

    st.set_page_config(
        page_title="Alaska Department of Snow Online Agent", page_icon="❄️"
    )
    st.header("Alaska Department of Snow Online Agent ❄️")
    st.badge(f"Session ID: {st.session_state.session_id}", color="blue")

    # Set the agent on load to the session
    st.session_state.graph = load_graph(st.session_state.session_id)

    # Initialize the message as empty on reload
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # When app refreshes reload the message from history
    for message in [
        message
        for message in st.session_state.messages
        if ("is_tool" not in message or not message["is_tool"])
    ]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user questions and trigger the agent to get response
    if user_question := st.chat_input(
        "Enter your question",
    ):
        # Appends the user message first to the history
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        graph: CompiledStateGraph = st.session_state.graph | message_chunk_to_str

        with st.chat_message("assistant"):
            # Stream the agent response
            response = st.write_stream(
                graph.stream(
                    {"messages": [HumanMessage(user_question)]},
                    stream_mode="messages",
                )
            )

            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
