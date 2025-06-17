from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_google_community import BigQueryVectorStore

from langgraph.graph import MessagesState

PROJECT_ID = "qwiklabs-gcp-03-7a8bdf6e2e2c"
LOCATION = "us"
DATASET = "AlaskaDept"
TABLE = "faqs"
TABLE_EMBEDDED = "faqs_embedded"

embedding = VertexAIEmbeddings(model_name="text-embedding-005", project=PROJECT_ID)

vector_store = BigQueryVectorStore(
    project_id=PROJECT_ID,
    dataset_name=DATASET,
    table_name=TABLE_EMBEDDED,
    location=LOCATION,
    embedding=embedding,
)

retriever = vector_store.as_retriever()

llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001",
    temperature=1,
    max_tokens=None,
    max_retries=2,
    stop=None,
)


class OverallState(MessagesState):
    input_guard_action: str
