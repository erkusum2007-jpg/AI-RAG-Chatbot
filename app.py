
import os
import streamlit as st
import numpy as np
import faiss
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="AI PDF RAG Chatbot", page_icon="🤖")
st.set_page_config(page_title="AI PDF RAG Chatbot", page_icon="🤖")
st.title("🤖 AI PDF RAG Chatbot")
with st.sidebar:st.markdown("---")
st.caption("Version 2.0")
st.title("🤖 AI RAG Chatbot")

st.markdown("---")

st.write("### 📂 Uploaded PDFs")

if "messages" in st.session_state:
        st.write(f"💬 Chat Messages: {len(st.session_state.messages)}")

if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.markdown("---")

st.info(
        """
        📌 Features

        ✅ Multiple PDF Support
        ✅ FAISS Search
        ✅ Groq Llama 3.3
        ✅ AI Chat
        """
    )
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

pdfs = st.file_uploader("📚 Upload one or more PDFs", type="pdf", accept_multiple_files=True)
if pdfs:
    st.sidebar.success(f"📄 PDFs Uploaded: {len(pdfs)}")
if pdfs:
    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

    chunks = [text[i:i+800] for i in range(0, len(text), 600)]
    emb = model.encode(chunks)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(np.array(emb))

    st.success("PDFs processed successfully ✅")
st.sidebar.write("### 📂 Uploaded Files")

for pdf in pdfs:
    st.sidebar.write(f"✅ {pdf.name}")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    q = st.chat_input("Ask a question about your PDFs...")
    if q:
        st.session_state.messages.append({"role":"user","content":q})
        with st.chat_message("user"):
            st.write(q)

        qemb = model.encode([q])
        _, ids = index.search(np.array(qemb), 5)
        context = "\n".join(chunks[i] for i in ids[0])
        st.caption(f"📄 Answer generated from {len(ids[0])} relevant text chunks.")
    
        prompt = f"""Use ONLY this context.

Context:
{context}

Question:
{q}

If the answer is not present, say you couldn't find it in the uploaded PDFs.
"""
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}]
        )
        ans = resp.choices[0].message.content
        st.session_state.messages.append({"role":"assistant","content":ans})
        with st.chat_message("assistant"):
            st.write(ans)
