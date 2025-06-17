from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from common import llm, OverallState

# guardrail prompt to check the question relevancy
guardrails_system = """
You are an intelligent assistant that determines whether a user question is related to the Alaska Department of Snow (ADS).

The Alaska Department of Snow (ADS) is responsible for snow-related public services in Alaska, including:
- Snowfall alerts and weather-related updates
- Road and highway conditions during snow events
- Plowing schedules and operations
- School and public service closures due to snow
- General service disruptions caused by winter weather

Your task is to assess the user’s question and output one of the following:
- "ok" — if the question is related to ADS responsibilities listed above, or if it is a common greeting (e.g., "hi", "hello", "good morning").
- "end" — if the question is unrelated to snow, weather services, or ADS operations.

Only return one of the two strings exactly: "ok" or "end". Do not include any explanation or extra text.
"""
guardrails_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            guardrails_system,
        ),
        (
            "human",
            ("{question}"),
        ),
    ]
)


class GuardrailsOutput(BaseModel):
    decision: Literal["ok", "end"] = Field(
        description="Decision on whether the question is related to Alaska Department of Snow"
    )


guardrails_chain = guardrails_prompt | llm.with_structured_output(GuardrailsOutput)


def guardrails(state: OverallState) -> OverallState:
    """
    Decides if the question is related to Alaska Department of Snow or not.
    """
    guardrails_output = guardrails_chain.invoke(
        {"question": state.get("messages")[-1].content}
    )
    if guardrails_output.decision == "end":
        return {
            "messages": [
                AIMessage(
                    "I'm sorry, but this question doesn't appear to relate to the Alaska Department of Snow. "
                    "I'm here to help with topics like snow removal, road conditions, school closures, and other ADS services. "
                    "Please feel free to ask about those!"
                )
            ],
            "input_guard_action": "end",
        }

    return {"input_guard_action": "ok"}


# Response guard

# guardrail prompt to check the response relevancy
response_guardrails_system = """
You are a response guard for an intelligent assistant that serves the Alaska Department of Snow (ADS).

Your job is to review AI-generated responses before they are shown to the user. Evaluate each response based on the following criteria:

1. **Relevance**: The response should pertain to snow-related public services in Alaska, such as:
   - Road and highway conditions
   - Snowfall forecasts and plowing schedules
   - School or government closures due to snow
   - Service disruptions caused by snow or ice
   - General snow safety, alerts, and response operations
   - Basic greetings and polite follow-ups are also acceptable

2. **Appropriateness**: The response must be safe, professional, and respectful. It should not include:
   - Inappropriate or offensive content
   - Fabricated or speculative information not grounded in context
   - Opinions, jokes, or unrelated commentary

3. **Clarity**: The response should be concise, helpful, and easy to understand.

Your task:
- If the response meets all criteria, output `"ok"`.
Only return one of these two strings: `"ok"` or `"reject"` — no explanations, summaries, or extra text.
"""
response_guardrails_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            response_guardrails_system,
        ),
        (
            "human",
            ("{response}"),
        ),
    ]
)


class ResponseGuardrailsOutput(BaseModel):
    decision: Literal["ok", "reject"] = Field(
        description="Decision on whether the response is related to Alaska Department of Snow"
    )


response_guardrails_chain = response_guardrails_prompt | llm.with_structured_output(
    ResponseGuardrailsOutput
)


def response_guard(state: dict) -> dict:
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage):
        return state  # Only check AI responses

    response_text = last_message.content.lower()

    guardrails_output = response_guardrails_chain.invoke({"response": response_text})

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

    return {  # Valid response
        "messages": state["messages"],
        "response_guard_action": "ok",
    }
