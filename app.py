"""Streamlit chat: grounded answer, source passages, and the reviewer's verdict."""
import streamlit as st

from src import config
from src.graph import answer

st.set_page_config(
    page_title="Two-Agent Docs Assistant",
    page_icon=":books:",
    layout="centered",
)

st.title("Two-Agent Grounded Q&A Assistant")
st.caption(
    "Researcher searches the Qdrant collection, Drafter answers from the retrieved "
    "passages, Reviewer verifies every claim and sends it back once if anything is "
    "unsupported. Corpus: official LangChain and Qdrant documentation."
)

try:
    config.require_env()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()


def render_sources(passages):
    with st.expander(f"Sources ({len(passages)} passages)"):
        for i, p in enumerate(passages, start=1):
            st.markdown(f"**[{i}] {p['source']} — {p['title'] or 'untitled'}**  ·  score {p['score']}")
            if p["url"]:
                st.markdown(f"[{p['url']}]({p['url']})")
            snippet = p["text"][:500]
            st.caption(snippet + ("..." if len(p["text"]) > 500 else ""))
            st.divider()


def render_verdict(verdict, reason, revisions):
    if verdict == "APPROVED":
        st.success(f"Reviewer verdict: APPROVED — {reason}")
    else:
        st.warning(f"Reviewer verdict: REJECTED — {reason}")
    if revisions:
        st.info(f"Handoff loop fired: draft sent back {revisions} time(s) for revision.")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        extras = message.get("extras")
        if extras:
            render_sources(extras["passages"])
            render_verdict(extras["verdict"], extras["reason"], extras["revisions"])

question = st.chat_input("Ask about LangChain or Qdrant...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Researcher → Drafter → Reviewer..."):
            try:
                result = answer(question)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong: {exc}")
                st.stop()

        st.markdown(result["draft"])
        render_sources(result["passages"])
        render_verdict(
            result.get("verdict", ""),
            result.get("reason", ""),
            result.get("revisions", 0),
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["draft"],
        "extras": {
            "passages": result["passages"],
            "verdict": result.get("verdict", ""),
            "reason": result.get("reason", ""),
            "revisions": result.get("revisions", 0),
        },
    })

with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "1. **Researcher** embeds your question and searches the remote Qdrant "
        "collection, returning the top passages with their source URLs.\n"
        "2. **Drafter** writes an answer using only those passages.\n"
        "3. **Reviewer** checks each claim against the passages. If anything is "
        "unsupported it sends the draft back to the Drafter — once.\n\n"
        "The Reviewer→Drafter edge is a conditional edge in LangGraph, which is "
        "what makes this a handoff loop rather than a straight pipeline."
    )
    st.subheader("Configuration")
    st.code(
        f"collection: {config.COLLECTION_NAME}\n"
        f"embeddings: {config.EMBEDDING_MODEL}\n"
        f"llm:        {config.LLM_MODEL}\n"
        f"top_k:      {config.TOP_K}\n"
        f"chunk:      {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}",
        language="text",
    )
