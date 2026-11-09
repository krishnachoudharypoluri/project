import re
from typing import TypedDict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.services.knowledge_base import knowledge_base
from app.services.orders import track_order


class AssistantState(TypedDict, total=False):
    message: str
    intent: str
    documents: list[Document]
    tool_result: str
    answer: str


def classify(state: AssistantState) -> dict[str, str]:
    message = state["message"].lower()
    if any(term in message for term in ("track", "where is my order", "order status")):
        intent = "order_tracking"
    elif any(term in message for term in ("return", "refund", "exchange")):
        intent = "returns"
    elif any(term in message for term in ("recommend", "suggest", "looking for", "best")):
        intent = "recommendations"
    elif any(term in message for term in ("compare", "difference", "versus", " vs ")):
        intent = "comparison"
    elif any(term in message for term in ("price", "cost", "sku", "model", "stock", "inventory", "available")):
        intent = "product_lookup"
    else:
        intent = "product_information"
    return {"intent": intent}


def retrieve(state: AssistantState) -> dict[str, list[Document]]:
    documents = knowledge_base.search(state["message"])
    result: dict[str, list[Document]] = {"documents": documents}
    if state.get("intent") == "order_tracking":
        match = re.search(r"\bSA-\d+\b", state["message"], re.IGNORECASE)
        if not match:
            result["tool_result"] = "Ask the customer for their order ID (for example, SA-1042)."
        else:
            order = track_order(match.group(0))
            result["tool_result"] = (
                f"Order {order['order_id']} is {order.get('status', 'status unavailable')}. "
                f"Estimated delivery: {order.get('estimated_delivery', 'not available')}."
                if order
                else f"No order was found for {match.group(0)}."
            )
    return result


def respond(state: AssistantState) -> dict[str, str]:
    settings = get_settings()
    context = "\n\n".join(document.page_content for document in state.get("documents", []))
    tool_result = state.get("tool_result", "")

    if settings.openai_api_key:
        model = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0)
        prompt = (
            "You are ShopAssist, a helpful e-commerce support assistant. Answer using only the supplied context "
            "and tool result. If the answer is not present, say what information is missing. Do not invent prices, "
            "inventory, policies, or order details.\n\n"
            f"Customer question: {state['message']}\n\nRetrieved context:\n{context or 'No matching documents.'}\n\n"
            f"Tool result:\n{tool_result or 'No tool was needed.'}"
        )
        response = model.invoke(prompt)
        answer = str(response.content)
    elif tool_result:
        answer = tool_result
    elif context:
        answer = f"Here is the information I found:\n\n{context}"
    else:
        answer = "I couldn't find that in the current product and policy information. Try a product name, SKU, or a more specific question."

    return {"answer": answer}


workflow = StateGraph(AssistantState)
workflow.add_node("classify", classify)
workflow.add_node("retrieve_and_use_tools", retrieve)
workflow.add_node("respond", respond)
workflow.add_edge(START, "classify")
workflow.add_edge("classify", "retrieve_and_use_tools")
workflow.add_edge("retrieve_and_use_tools", "respond")
workflow.add_edge("respond", END)
assistant_graph = workflow.compile()


def answer_question(message: str) -> AssistantState:
    return assistant_graph.invoke({"message": message})