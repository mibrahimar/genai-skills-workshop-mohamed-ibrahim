import io
from langchain_core.runnables.graph import MermaidDrawMethod

from IPython.display import Image
from PIL import Image
from agent import get_agent

agent = get_agent()

# used to visualize the langgraph flow image
image = Image.open(
    io.BytesIO(
        agent.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)
image.show()
