# 🤖 Task 1: Dynamic Knowledge Base Customer Service Chatbot

A production-ready Customer Service Chatbot built using **Retrieval-Augmented Generation (RAG)** 
that supports dynamic knowledge base updates in real-time — without any model retraining.

---

## 📌 Project Overview

This chatbot answers customer queries by retrieving relevant information from a 
knowledge base (FAQ dataset) and generating grounded responses using Google Gemini 1.5 Flash.
New knowledge can be added at any time through CSV upload, TXT upload, URL scraping, 
or manual entry — and becomes immediately available to the chatbot.

---

## 🎯 Key Features

- ✅ **RAG Architecture** — answers grounded in verified documents, zero hallucination
- ✅ **Dynamic KB Updates** — add new knowledge without rebuilding the index
- ✅ **Auto-Watcher** — background thread monitors `sources/` folder every 5 minutes
- ✅ **4 Ingestion Methods** — CSV, TXT, URL scraping, manual Q&A entry
- ✅ **FAISS Vector Database** — fast semantic similarity search
- ✅ **Persistent Index** — FAISS index saved to disk, reloaded on restart
- ✅ **Thread-Safe** — concurrent read/write protection via threading.Lock()
- ✅ **Two-Tab Streamlit UI** — Chat + Knowledge Base Manager

---

## 🛠️ Technology Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10 | Core language |
| LangChain | 0.0.339 | RAG orchestration |
| FAISS | 1.7.4 | Vector similarity search |
| sentence-transformers | all-MiniLM-L6-v2 | Text embeddings |
| Google Gemini 1.5 Flash | latest | LLM answer generation |
| Streamlit | 1.28.0 | Web application UI |
| Pandas | latest | CSV data processing |
| BeautifulSoup4 | latest | URL scraping |

---

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/task1-dynamic-kb-chatbot.git
cd task1-dynamic-kb-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Create .env file
Create a `.env` file in the project root:

Get a free API key at: https://aistudio.google.com

### 5. Run the application
```bash
streamlit run main.py
```

Open browser at: `http://localhost:8501`

---

## 🚀 How to Use

### Chat Tab
1. Type your question in the input box at the top
2. Click **Send** to get an answer
3. The chatbot retrieves relevant FAQs and generates a grounded response

### Manage KB Tab
Add new knowledge using any of these methods:

| Method | How |
|---|---|
| 📄 Upload CSV | Upload a CSV with `prompt` and `response` columns |
| 📝 Upload TXT | Upload any text file — auto-chunked into 400-char pieces |
| 🌐 Scrape URL | Enter a public URL — page content scraped and indexed |
| ✏️ Manual Entry | Type Q&A pairs directly in the UI |

### Auto-Watcher
Drop any `.csv` or `.txt` file into the `sources/` folder.
The watcher checks every 5 minutes and auto-ingests new files — no manual action needed.

---

## 📊 Dataset

**NullClass Data Science Bootcamp FAQ Dataset**
- 76 verified Q&A pairs
- Topics: course fees, prerequisites, curriculum, placement assistance,
  refund policy, technical support, tools covered
- Format: CSV with `prompt` and `response` columns

---

## 🔒 Security Notes

- API key stored in `.env` file — never committed to GitHub
- `.env` is listed in `.gitignore`
- No user data stored or transmitted externally beyond Gemini API calls

---

## 🔮 Future Enhancements

- [ ] Multi-user authentication
- [ ] Conversation memory across sessions
- [ ] Support for PDF ingestion
- [ ] Analytics dashboard (most asked questions, unanswered rate)
- [ ] Docker deployment
- [ ] Migrate to Pinecone for cloud-hosted vector search

---

## 👨‍💻 Author

**Advait Hegde**  

Internship Project — NullClass

---
