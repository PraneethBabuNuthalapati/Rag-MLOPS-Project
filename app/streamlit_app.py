import streamlit as st
import requests
import time

BACKEND_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="RAG Document Q&A", layout="wide")

# -----------------------------
# SESSION STATE
# -----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "files" not in st.session_state:
    st.session_state.files = []


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("📄 RAG Q&A")

    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Upload"):
        if uploaded_files:
            files_payload = [
                ("files", (file.name, file.getvalue(), "application/pdf"))
                for file in uploaded_files
            ]

            with st.spinner("Uploading..."):
                response = requests.post(f"{BACKEND_URL}/upload", files=files_payload)

            if response.status_code == 200:
                data = response.json()
                st.session_state.session_id = data["session_id"]
                st.session_state.files = [f.name for f in uploaded_files]
                st.session_state.chat_history = []
                st.success("Uploaded successfully!")
                st.rerun()
            else:
                st.error("Upload failed")
        else:
            st.warning("Please upload at least one file")

    st.divider()

    if st.session_state.session_id:
        st.success("Session Active ✅")

        st.write("📂 Files:")
        for f in st.session_state.files:
            st.write(f"• {f}")

        if st.button("🔄 New Session"):
            st.session_state.session_id = None
            st.session_state.chat_history = []
            st.session_state.files = []
            st.rerun()


# -----------------------------
# MAIN CHAT AREA
# -----------------------------
st.title("💬 Ask Questions About Your Documents")

if not st.session_state.session_id:
    st.info("Upload documents to start asking questions.")
    st.stop()


# -----------------------------
# DISPLAY CHAT
# -----------------------------
for chat in st.session_state.chat_history:
    question = chat.get("q") or chat.get("query") or chat.get("question") or ""
    answer = chat.get("a") or chat.get("answer") or chat.get("response") or ""
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        st.write(answer)

        if chat["sources"]:
            with st.expander("📎 Sources"):
                for src in chat["sources"]:
                    if isinstance(src, dict):
                        st.markdown(f"**📄 {src.get('file','unknown')}**")
                        st.caption(src.get("snippet", ""))
                        st.divider()
                    else:
                        st.write(src)


# -----------------------------
# INPUT BOX (CHAT STYLE)
# -----------------------------
query = st.chat_input("Ask something about your documents...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={
                    "query": query,
                    "session_id": st.session_state.session_id,
                    "history": st.session_state.chat_history
                }
            )

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer found.")
            sources = data.get("sources", [])

            # st.write(answer)
            placeholder = st.empty()
            typed_text = ""

            for char in answer:
                typed_text += char
                placeholder.markdown(typed_text)
                time.sleep(0.005)

            if sources:
                with st.expander("📎 Sources"):
                    for src in sources:
                        if isinstance(src, dict):
                            st.markdown(f"**📄 {src.get('file','unknown')}**")
                            st.caption(src.get("snippet", ""))
                            st.divider()
                        else:
                            st.write(src)

            # save history
            st.session_state.chat_history.append({
                "q": query,
                "a": answer,
                "sources": sources
            })

        else:
            st.error("Query failed")