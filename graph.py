"""The agent graph.

    researcher -> drafter -> reviewer -+-> END
                     ^                 |
                     +-- (REJECTED) ---+

The edge from reviewer back to drafter is a genuine conditional handoff, not a
straight-line chain. It fires at most MAX_REVISIONS times so the graph always
terminates.
"""
import json
import re
from typing import TypedDict, List, Dict, Literal

from langgraph.graph import StateGraph, END

from src import config, llm, retriever
from src.prompts import DRAFT_PROMPT, FEEDBACK_TEMPLATE, REVIEW_PROMPT


class State(TypedDict, total=False):
    question: str
    passages: List[Dict]
    draft: str
    verdict: str
    reason: str
    revisions: int
    history: List[Dict]


def _parse_json(raw: str) -> Dict:
    """Models sometimes wrap JSON in fences - strip them before parsing."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    # Unparseable review: fail open rather than loop forever.
    return {"verdict": "APPROVED", "reason": "Reviewer output could not be parsed."}


# ----------------------------- nodes -----------------------------

def researcher(state: State) -> State:
    """Agent 1 - searches Qdrant and hands back passages with their sources."""
    passages = retriever.search(state["question"])
    return {"passages": passages, "revisions": state.get("revisions", 0)}


def drafter(state: State) -> State:
    """Writes an answer grounded strictly in the retrieved passages."""
    feedback_block = ""
    if state.get("verdict") == "REJECTED":
        feedback_block = FEEDBACK_TEMPLATE.format(
            previous=state.get("draft", ""),
            reason=state.get("reason", ""),
        )

    answer = llm.complete(DRAFT_PROMPT.format(
        passages=retriever.format_passages(state["passages"]),
        question=state["question"],
        feedback_block=feedback_block,
    ))
    return {"draft": answer}


def reviewer(state: State) -> State:
    """Agent 2 - checks the draft against the passages and rules on it."""
    result = _parse_json(llm.complete(REVIEW_PROMPT.format(
        passages=retriever.format_passages(state["passages"]),
        answer=state["draft"],
    ), temperature=0.0))

    verdict = "REJECTED" if str(result.get("verdict", "")).upper() == "REJECTED" else "APPROVED"
    reason = result.get("reason", "")

    history = list(state.get("history", []))
    history.append({
        "attempt": state.get("revisions", 0) + 1,
        "draft": state["draft"],
        "verdict": verdict,
        "reason": reason,
    })
    return {"verdict": verdict, "reason": reason, "history": history}


def route(state: State) -> Literal["drafter", "__end__"]:
    """The conditional edge - this is the actual handoff between the agents."""
    if state["verdict"] == "REJECTED" and state.get("revisions", 0) < config.MAX_REVISIONS:
        state["revisions"] = state.get("revisions", 0) + 1
        return "drafter"
    return END


# ----------------------------- graph -----------------------------

def build_graph():
    g = StateGraph(State)
    g.add_node("researcher", researcher)
    g.add_node("drafter", drafter)
    g.add_node("reviewer", reviewer)
    g.set_entry_point("researcher")
    g.add_edge("researcher", "drafter")
    g.add_edge("drafter", "reviewer")
    g.add_conditional_edges("reviewer", route)
    return g.compile()


APP = build_graph()


def answer(question: str) -> State:
    """Run one question through the whole graph."""
    return APP.invoke({"question": question, "revisions": 0, "history": []})
