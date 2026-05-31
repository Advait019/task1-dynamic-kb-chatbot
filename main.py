"""
main.py — Dynamic Knowledge Base Chatbot (Task 1)
"""

import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

from kb_manager import KnowledgeBaseManager, SOURCES_DIR

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dynamic KB Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] { font-size:16px; font-weight:600; padding:12px 24px; }
    .metric-box { background:#f0f4ff; border-radius:10px; padding:16px;
                  text-align:center; margin:4px; }
    .metric-box h2 { color:#1a56db; margin:0; font-size:2rem; }
    .metric-box p  { color:#555; margin:0; font-size:.85rem; }
    .source-card { background:#fff; border:1px solid #e0e7ff;
                   border-left:4px solid #4f46e5; border-radius:8px;
                   padding:10px 14px; margin:6px 0; }
    .source-card small { color:#6b7280; }
    .chat-user { background:#e8f0fe; padding:10px 14px;
                 border-radius:12px; margin:6px 0; }
    .chat-bot  { background:#f0fdf4; padding:10px 14px;
                 border-radius:12px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)


# ── Singleton manager ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading Knowledge Base …")
def get_manager():
    mgr = KnowledgeBaseManager()
    mgr.start_watcher(interval_minutes=5)
    return mgr


mgr = get_manager()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"       not in st.session_state:
    st.session_state.messages = []
if "chain"          not in st.session_state:
    st.session_state.chain = None
if "manual_entries" not in st.session_state:
    st.session_state.manual_entries = [{"prompt": "", "response": ""}]
if "last_input"     not in st.session_state:
    st.session_state.last_input = ""


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    stats = mgr.get_stats()
    st.sidebar.title("📊 KB Status")
    st.sidebar.markdown("---")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Total Docs", stats["total_docs"])
    c2.metric("Sources",    len(stats["sources"]))
    badge = "🟢 Running" if stats["watcher_on"] else "⚫ Stopped"
    st.sidebar.markdown(f"**Auto-Watcher:** {badge}")
    st.sidebar.caption(f"Last updated: {stats['last_updated']}")

    if st.sidebar.button("🔄 Force Check sources/ Now"):
        with st.sidebar.spinner("Scanning …"):
            n = mgr._watch_loop_once()
        if n:
            st.sidebar.success(f"Added {n} new docs!")
            st.session_state.chain = None
        else:
            st.sidebar.info("No new files found.")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📂 Drop files here to auto-ingest:**")
    st.sidebar.code(str(SOURCES_DIR))
    st.sidebar.caption("Supported: `.csv` (prompt/response columns) or `.txt`")


render_sidebar()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_manage = st.tabs(["🤖  Chat History", "📚  Manage Knowledge Base"])


# =============================================================================
# TAB 1 — CHAT HISTORY
# =============================================================================
with tab_chat:
    st.title("🤖 Customer Service Chatbot")
    st.caption(
        "Powered by **Retrieval-Augmented Generation** — "
        "answers are grounded in the live knowledge base."
    )

    col_refresh, col_clear = st.columns([2, 1])
    with col_refresh:
        if st.button("🔃 Refresh Chain"):
            st.session_state.chain = None
            st.success("Chain will rebuild on your next question.")
    with col_clear:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.last_input = ""
            st.rerun()

    st.markdown("---")

    if not st.session_state.messages:
        st.info("👋 Type your question below and click **Send** to get started.")
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🧑 <b>You:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bot">🤖 <b>Assistant:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("### 💬 Ask a Question")
    col_input, col_send = st.columns([6, 1])
    with col_input:
        user_input = st.text_input(
            label="question",
            placeholder="e.g. What is the fee for the Data Science bootcamp?",
            label_visibility="collapsed",
            key="question_input",
        )
    with col_send:
        send_clicked = st.button("Send ➤", use_container_width=True)

    if send_clicked and user_input.strip() and user_input.strip() != st.session_state.last_input:
        query = user_input.strip()
        st.session_state.last_input = query
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking …"):
            try:
                if st.session_state.chain is None:
                    st.session_state.chain = mgr.get_qa_chain()

                result  = st.session_state.chain({"query": query})
                answer  = result.get("result", "No answer returned.")
                sources = result.get("source_documents", [])

            except RuntimeError as e:
                answer  = (
                    f"⚠️ {e}\n\n"
                    "👉 Go to the **Manage KB** tab or click **Force Check** in the sidebar."
                )
                sources = []
            except Exception as e:
                answer  = f"⚠️ Error: {e}"
                sources = []

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


# =============================================================================
# TAB 2 — MANAGE KB
# =============================================================================
with tab_manage:
    st.title("📚 Knowledge Base Manager")
    st.caption("Dynamically expand the chatbot's knowledge — **no retraining needed**.")

    stats = mgr.get_stats()
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="metric-box"><h2>{stats["total_docs"]}</h2><p>Total Documents</p></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-box"><h2>{len(stats["sources"])}</h2><p>Ingested Sources</p></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="metric-box"><h2>{"✅" if stats["watcher_on"] else "⏸️"}</h2><p>Auto-Watcher</p></div>',
        unsafe_allow_html=True,
    )

    if stats["sources"]:
        st.markdown("**Ingested sources (most recent first):**")
        for src in reversed(stats["sources"]):
            st.markdown(
                f'<div class="source-card">📄 <b>{src["name"]}</b> &nbsp;|&nbsp; '
                f'{src["docs_added"]} docs added<br>'
                f'<small>🕐 {src["timestamp"]}</small></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("➕ Add New Knowledge")

    method = st.radio(
        "Choose input method:",
        ["📄 Upload CSV", "📝 Upload TXT", "🌐 Scrape URL", "✏️ Manual Entry"],
        horizontal=True,
    )

    # ── CSV ───────────────────────────────────────────────────────────────────
    if method == "📄 Upload CSV":
        st.markdown("CSV must have **`prompt`** and **`response`** columns.")
        uploaded = st.file_uploader("Choose CSV", type=["csv"])
        if uploaded:
            df_prev = pd.read_csv(uploaded)
            st.dataframe(df_prev.head(5), use_container_width=True)
            st.caption(f"{len(df_prev)} rows detected")
            if st.button("📥 Add to Knowledge Base"):
                with st.spinner("Indexing …"):
                    with tempfile.NamedTemporaryFile(
                        suffix=".csv", delete=False, mode="wb"
                    ) as tf:
                        tf.write(uploaded.getbuffer())
                        tmp = tf.name
                    n = mgr.add_from_csv(tmp, source_name=uploaded.name)
                if n:
                    st.success(f"✅ Added {n} docs from **{uploaded.name}**")
                    st.session_state.chain = None
                    st.rerun()
                else:
                    st.error("No documents added — check CSV format.")
        with st.expander("Expected CSV format"):
            st.code(
                "prompt,response\n"
                "What is the fee?,The fee is ₹15000.\n"
                "Is there a refund?,Yes, within 7 days."
            )

    # ── TXT ───────────────────────────────────────────────────────────────────
    elif method == "📝 Upload TXT":
        txt_file = st.file_uploader("Choose TXT file", type=["txt"])
        if txt_file:
            preview = txt_file.read().decode("utf-8", errors="ignore")
            st.text_area(
                "Preview:",
                preview[:600] + ("…" if len(preview) > 600 else ""),
                height=140,
            )
            txt_file.seek(0)
            if st.button("📥 Add to Knowledge Base"):
                with st.spinner("Chunking and indexing …"):
                    with tempfile.NamedTemporaryFile(
                        suffix=".txt", delete=False, mode="wb"
                    ) as tf:
                        tf.write(txt_file.getbuffer())
                        tmp = tf.name
                    n = mgr.add_from_txt(tmp)
                if n:
                    st.success(f"✅ Added {n} chunks from **{txt_file.name}**")
                    st.session_state.chain = None
                    st.rerun()
                else:
                    st.error("No content extracted.")

    # ── URL ───────────────────────────────────────────────────────────────────
    elif method == "🌐 Scrape URL":
        url = st.text_input("URL:", placeholder="https://example.com/faq")
        if url and st.button("🌐 Scrape & Add"):
            with st.spinner(f"Scraping {url} …"):
                n = mgr.add_from_url(url)
            if n:
                st.success(f"✅ Added {n} chunks from URL")
                st.session_state.chain = None
                st.rerun()
            else:
                st.error("Could not extract content from URL.")

    # ── Manual Entry ──────────────────────────────────────────────────────────
    elif method == "✏️ Manual Entry":
        updated = []
        for i, entry in enumerate(st.session_state.manual_entries):
            st.markdown(f"**Entry {i+1}**")
            cq, ca = st.columns(2)
            q = cq.text_input(
                f"Question {i+1}", value=entry["prompt"], key=f"q_{i}"
            )
            a = ca.text_area(
                f"Answer {i+1}", value=entry["response"], key=f"a_{i}", height=80
            )
            updated.append({"prompt": q, "response": a})
        st.session_state.manual_entries = updated

        c1, c2 = st.columns([1, 2])
        if c1.button("➕ Add row"):
            st.session_state.manual_entries.append({"prompt": "", "response": ""})
            st.rerun()

        label = st.text_input("Source label:", value="manual_entry")
        if c2.button("💾 Save to Knowledge Base"):
            valid = [
                e for e in st.session_state.manual_entries
                if e["prompt"] and e["response"]
            ]
            if not valid:
                st.warning("Fill in at least one complete Q&A pair.")
            else:
                with st.spinner("Indexing …"):
                    n = mgr.add_from_text(valid, source_name=label)
                st.success(f"✅ Added {n} Q&A pairs")
                st.session_state.manual_entries = [{"prompt": "", "response": ""}]
                st.session_state.chain = None
                st.rerun()

    # ── Watcher info ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⏱️ Auto-Watcher")
    st.info(
        f"Watching **`{SOURCES_DIR}`** every 5 minutes. "
        "Drop a `.csv` or `.txt` there — it will be auto-ingested "
        "with no manual action needed."
    )
    with st.expander("How it works"):
        st.markdown(
            "```\n"
            "Every 5 minutes:\n"
            "  1. Scan sources/ folder\n"
            "  2. Compare filenames to kb_metadata.json\n"
            "  3. For each NEW file:\n"
            "       .csv → pandas read → embed rows  → FAISS.merge_from()\n"
            "       .txt → chunk (400 chars) → embed → FAISS.merge_from()\n"
            "  4. Save updated FAISS index to disk\n"
            "  5. Update kb_metadata.json\n"
            "```\n"
            "The chatbot uses the updated index on the "
            "**very next question** — no restart needed."
        )