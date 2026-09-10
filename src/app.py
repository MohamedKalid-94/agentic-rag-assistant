import streamlit as st
import os
from vector_store import rebuild_vector_store
from graph import graph

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🤖")

st.title("🤖 Agentic RAG Assistant")
st.caption("Ask questions about your uploaded documents — powered by LangGraph, hybrid search, and self-correcting retrieval.")

# --- Sidebar: document upload + rebuild ---
with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("data/documents", exist_ok=True)
        for file in uploaded_files:
            save_path = os.path.join("data/documents", file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to data/documents/")

    st.divider()

    existing_files = os.listdir("data/documents") if os.path.exists("data/documents") else []
    st.write(f"**Current documents ({len(existing_files)}):**")
    for f in existing_files:
        st.write(f"- {f}")

    if st.button("🔄 Rebuild vector store", type="primary"):
        with st.spinner("Rebuilding: loading, chunking, embedding, storing..."):
            try:
                chunk_count = rebuild_vector_store()
                st.success(f"Rebuilt! Stored {chunk_count} chunks.")
            except Exception as e:
                st.error(f"Rebuild failed: {e}")

# --- Main: question + answer ---
question = st.text_input("Ask a question about your documents:")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        try:
            result = graph.invoke({
                "question": question,
                "original_question": question,
                "filters": {},
                "retrieved_chunks": [],
                "is_relevant": False,
                "attempts": 0,
                "answer": "",
                "is_grounded": False,
                "generation_attempts": 0
            })

            st.markdown("### Answer")
            st.write(result["answer"])

            with st.expander("🔍 Reasoning trace"):
                st.write(f"**Filters extracted:** {result['filters']}")
                st.write(f"**Retrieval attempts:** {result['attempts']}")
                st.write(f"**Relevant:** {result['is_relevant']}")
                st.write(f"**Grounded:** {result['is_grounded']}")
                st.write(f"**Generation attempts:** {result['generation_attempts']}")

                st.write("**Retrieved chunks:**")
                for chunk_id, doc, score, metadata in result["retrieved_chunks"]:
                    st.write(f"- `{chunk_id}` — {metadata.get('section')} (Month {metadata.get('month')}, Week {metadata.get('week')})")

        except Exception as e:
            st.error(f"Something went wrong: {e}")