"""LangGraph wiring: understand -> plan -> execute (loop) -> verify -> complete, with one repair pass on failed verification."""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import MAX_REPLANS, complete_node, execute_node, plan_node, understand_node, verify_node
from agent.state import AgentState


def route_after_execute(state: AgentState) -> Literal["execute", "verify"]:
    """Loop back to execute while plan steps remain (including retries of a failed step), else verify."""
    if state["current_step"] < len(state["plan"]):
        return "execute"
    return "verify"


def route_after_verify(state: AgentState) -> Literal["plan", "complete"]:
    """If verification failed and a repair pass is still allowed, go back to plan with the findings."""
    if not state.get("verified") and state.get("replans", 0) < MAX_REPLANS:
        return "plan"
    return "complete"


def build_graph() -> CompiledStateGraph:
    """Build and compile the five-node agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("understand", understand_node)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)
    graph.add_node("complete", complete_node)

    graph.add_edge(START, "understand")
    graph.add_edge("understand", "plan")
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges("execute", route_after_execute, {"execute": "execute", "verify": "verify"})
    graph.add_conditional_edges("verify", route_after_verify, {"plan": "plan", "complete": "complete"})
    graph.add_edge("complete", END)
    return graph.compile()
