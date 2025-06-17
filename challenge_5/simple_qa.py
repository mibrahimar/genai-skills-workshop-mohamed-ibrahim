from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
from langchain_google_community import BigQueryVectorStore

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import streamlit as st

PROJECT_ID = "qwiklabs-gcp-03-7a8bdf6e2e2c"
LOCATION = "us"
DATASET = "AlaskaDept"
TABLE = "faqs"
TABLE_EMBEDDED = "faqs_embedded"

embedding = VertexAIEmbeddings(model_name="text-embedding-005", project=PROJECT_ID)

store = BigQueryVectorStore(
    project_id=PROJECT_ID,
    dataset_name=DATASET,
    table_name=TABLE_EMBEDDED,
    location=LOCATION,
    embedding=embedding,
)

# See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
prompt = hub.pull("langchain-ai/retrieval-qa-chat")
llm = VertexAI(model_name="gemini-2.0-flash")

qa_chain = (
    {
        "context": store.as_retriever(),
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# print(qa_chain.invoke("what is 1 + 1?"))
# print(qa_chain.invoke("When was the Alaska Department of Snow established"))


def main():
    st.set_page_config(
        page_title="Alaska Department of Snow Online Agent", page_icon="❄️"
    )
    st.header("Alaska Department of Snow Online Agent ❄️")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if question := st.chat_input(
        "Enter your question",
    ):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            content = qa_chain.invoke(question)
            st.markdown(content)
            st.session_state.messages.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    main()
