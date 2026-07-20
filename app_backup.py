import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment
load_dotenv()

# Groq API
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Page setup
st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI PDF Chatbot (RAG)")
st.write("Upload your PDF and ask questions")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

model = load_embedding_model()


# Read PDF
def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# Split text
def chunk_text(text, size=500):
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])
    return chunks


# Create FAISS index
def create_index(chunks):

    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index, embeddings


# Ask Llama
def ask_llama(context, question):

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Answer only from the given document context."
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ]
    )

    return response.choices[0].message.content


# Upload PDF
pdf = st.file_uploader(
    "Upload PDF",
    type="pdf"
)


if pdf:

    text = extract_text(pdf)

    chunks = chunk_text(text)

    index, embeddings = create_index(chunks)

    st.success("PDF processed successfully ✅")


    question = st.text_input(
        "Ask your question"
    )


    if question:

        q_embedding = model.encode(
            [question]
        )

        distance, result = index.search(
            q_embedding,
            k=3
        )

        context = "\n".join(
            [chunks[i] for i in result[0]]
        )

        answer = ask_llama(
            context,
            question
        )

        st.subheader("Answer:")
        st.write(answer)