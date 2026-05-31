"""
kb_manager.py — Dynamic Knowledge Base Manager for the Customer-Service RAG Chatbot
Tested and verified working.
"""

import os
import json
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load .env from the same folder as this file
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
FAISS_PATH    = BASE_DIR / "faiss_index"
METADATA_FILE = BASE_DIR / "kb_metadata.json"
SOURCES_DIR   = BASE_DIR / "sources"
DATASET_CSV   = BASE_DIR / "dataset" / "dataset.csv"

SOURCES_DIR.mkdir(exist_ok=True)

# Gemini model for RAG answers (override via GEMINI_MODEL in .env)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

# ── Embeddings (loaded once at import) ────────────────────────────────────────
print("[KBManager] Loading embedding model ...")
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("[KBManager] Embedding model ready.")


# ─────────────────────────────────────────────────────────────────────────────
class KnowledgeBaseManager:

    def __init__(self):
        self._lock           = threading.Lock()
        self._index          = None
        self._meta           = self._load_metadata()
        self._watcher_thread = None
        self._watcher_stop   = threading.Event()

        if (FAISS_PATH / "index.faiss").exists():
            print("[KBManager] Loading existing FAISS index ...")
            self._index = FAISS.load_local(str(FAISS_PATH), EMBEDDINGS)
            print(f"[KBManager] Index loaded — {self._meta.get('total_docs', '?')} docs")
        else:
            print("[KBManager] No existing index — building from base dataset ...")
            self.add_from_csv(str(DATASET_CSV), source_name="base_dataset.csv")

    # ── Metadata ──────────────────────────────────────────────────────────────
    def _load_metadata(self) -> dict:
        if METADATA_FILE.exists():
            return json.loads(METADATA_FILE.read_text())
        return {"sources": [], "total_docs": 0, "last_updated": None}

    def _save_metadata(self):
        METADATA_FILE.write_text(json.dumps(self._meta, indent=2))

    def _record_source(self, source_name: str, doc_count: int):
        entry = {
            "name":       source_name,
            "docs_added": doc_count,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._meta["sources"] = [
            s for s in self._meta["sources"] if s["name"] != source_name
        ]
        self._meta["sources"].append(entry)
        self._meta["total_docs"] = self._meta.get("total_docs", 0) + doc_count
        self._meta["last_updated"] = entry["timestamp"]
        self._save_metadata()

    # ── Core: add documents to FAISS ─────────────────────────────────────────
    def _add_documents(self, docs: list, source_name: str) -> int:
        if not docs:
            print(f"[KBManager] No documents to add from '{source_name}'")
            return 0

        with self._lock:
            if self._index is None:
                self._index = FAISS.from_documents(docs, EMBEDDINGS)
            else:
                new_store = FAISS.from_documents(docs, EMBEDDINGS)
                self._index.merge_from(new_store)
            self._index.save_local(str(FAISS_PATH))

        self._record_source(source_name, len(docs))
        print(f"[KBManager] Added {len(docs)} docs from '{source_name}'")
        return len(docs)

    # ── Public ingestion methods ──────────────────────────────────────────────
    def add_from_csv(self, filepath: str, source_name: str = None) -> int:
        source_name = source_name or Path(filepath).name
        print(f"[KBManager] Loading CSV: {filepath}")
        try:
            import pandas as pd
            # latin-1 handles special characters in the dataset
            df = pd.read_csv(
                filepath,
                on_bad_lines="skip",
                engine="python",
                encoding="latin-1",
            )
            df.columns = df.columns.str.strip()

            if "prompt" not in df.columns or "response" not in df.columns:
                print(f"[KBManager] CSV must have 'prompt' and 'response' columns. Found: {list(df.columns)}")
                return 0

            df = df.dropna(subset=["prompt", "response"])
            df["prompt"]   = df["prompt"].astype(str).str.strip()
            df["response"] = df["response"].astype(str).str.strip()
            df = df[(df["prompt"] != "") & (df["response"] != "")]

            docs = [
                Document(
                    page_content=f"prompt: {row['prompt']}\nresponse: {row['response']}",
                    metadata={"source": row["prompt"]},
                )
                for _, row in df.iterrows()
            ]
            print(f"[KBManager] Loaded {len(docs)} valid rows from CSV")
            return self._add_documents(docs, source_name)
        except Exception as e:
            print(f"[KBManager] CSV error — {e}")
            traceback.print_exc()
            return 0

    def add_from_text(self, text_entries: list, source_name: str) -> int:
        docs = [
            Document(
                page_content=f"prompt: {e['prompt']}\nresponse: {e['response']}",
                metadata={"source": source_name},
            )
            for e in text_entries
            if e.get("prompt") and e.get("response")
        ]
        return self._add_documents(docs, source_name)

    def add_from_url(self, url: str) -> int:
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            raw_text = soup.get_text(separator="\n", strip=True)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks   = splitter.create_documents([raw_text], metadatas=[{"source": url}])
            return self._add_documents(chunks, source_name=url)
        except Exception as e:
            print(f"[KBManager] URL error — {e}")
            return 0

    def add_from_txt(self, filepath: str) -> int:
        source_name = Path(filepath).name
        try:
            loader   = TextLoader(filepath, encoding="utf-8")
            raw_docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
            docs     = splitter.split_documents(raw_docs)
            return self._add_documents(docs, source_name)
        except Exception as e:
            print(f"[KBManager] TXT error — {e}")
            return 0

    # ── Auto-watcher ─────────────────────────────────────────────────────────
    def _watch_loop(self, interval_seconds: int):
        print(f"[KBManager] Watcher started — checking every {interval_seconds}s")
        while not self._watcher_stop.is_set():
            self._watch_loop_once()
            time.sleep(interval_seconds)

    def _watch_loop_once(self) -> int:
        known = {s["name"] for s in self._meta.get("sources", [])}
        added = 0
        try:
            for fp in SOURCES_DIR.iterdir():
                if fp.name in known or fp.name.startswith("."):
                    continue
                print(f"[KBManager] New file found: {fp.name}")
                if fp.suffix.lower() == ".csv":
                    added += self.add_from_csv(str(fp), source_name=fp.name)
                elif fp.suffix.lower() == ".txt":
                    added += self.add_from_txt(str(fp))
                else:
                    print(f"[KBManager] Skipped unsupported file: {fp.name}")
        except Exception:
            traceback.print_exc()
        return added

    def start_watcher(self, interval_minutes: int = 5):
        if self._watcher_thread and self._watcher_thread.is_alive():
            print("[KBManager] Watcher already running")
            return
        self._watcher_stop.clear()
        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            args=(interval_minutes * 60,),
            daemon=True,
        )
        self._watcher_thread.start()

    def stop_watcher(self):
        self._watcher_stop.set()

    # ── QA Chain ─────────────────────────────────────────────────────────────
    def get_qa_chain(self):
        """
        Returns a plain Python callable mimicking RetrievalQA interface.
        Uses google.genai SDK directly — no LangChain LLM wrappers.
        NOTE: google-generativeai is deprecated; uses google.genai (new SDK).
        """
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

        # Use the new google.genai SDK
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)

            def call_gemini(prompt: str) -> str:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.1),
                )
                return response.text

        except ImportError:
            # Fallback to old SDK if new one not installed
            import google.generativeai as genai_old
            genai_old.configure(api_key=api_key)
            model = genai_old.GenerativeModel(
                GEMINI_MODEL,
                generation_config={"temperature": 0.1},
            )

            def call_gemini(prompt: str) -> str:
                response = model.generate_content(prompt)
                return response.text

        with self._lock:
            if self._index is None:
                raise RuntimeError("Knowledge base is empty. Add documents first.")
            retriever = self._index.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4},
            )

        def chain(inputs: dict) -> dict:
            query = inputs.get("query", "")

            try:
                docs = retriever.get_relevant_documents(query)
            except Exception as e:
                docs = []
                print(f"[Chain] Retrieval error: {e}")

            context = (
                "\n\n".join(doc.page_content for doc in docs)
                if docs else "No relevant documents found."
            )

            prompt = (
                "Given the following context and a question, generate an answer "
                "based on this context only.\n"
                "Provide as much detail as possible from the context without making up information.\n"
                "If the answer is not found in the context, kindly state "
                "\"I don't know.\" Don't try to make up an answer.\n\n"
                f"CONTEXT: {context}\n\n"
                f"QUESTION: {query}"
            )

            try:
                answer = call_gemini(prompt)
            except Exception as e:
                answer = f"Gemini API error: {e}"

            return {"result": answer, "source_documents": docs}

        return chain

    # ── Stats ─────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        return {
            "total_docs":   self._meta.get("total_docs", 0),
            "last_updated": self._meta.get("last_updated", "Never"),
            "sources":      self._meta.get("sources", []),
            "watcher_on":   bool(self._watcher_thread and self._watcher_thread.is_alive()),
        }
