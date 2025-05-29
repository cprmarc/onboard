import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from pathlib import Path
import glob
import os

st.set_page_config(page_title="AI Onboarding Chat", layout="wide")
st.title("💬 AI-alapú Onboarding Chat Asszisztens")

# OpenAI API kulcs
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("⚠️ Add meg az OpenAI API kulcsot a működéshez.")
    st.stop()

# Session state a beszélgetéshez
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Dokumentumok betöltése háttérből
DOCUMENT_DIR = "documents/"  # Ebbe a mappába töltsd a fájlokat manuálisan
if not os.path.exists(DOCUMENT_DIR):
    st.warning(f"⚠️ A '{DOCUMENT_DIR}' mappa nem található. Hozd létre és tölts bele fájlokat.")
    st.stop()

documents = []
for filepath in glob.glob(f"{DOCUMENT_DIR}/*"):
    path = Path(filepath)
    if path.suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix == ".txt":
        loader = TextLoader(str(path))
    elif path.suffix == ".html":
        loader = UnstructuredHTMLLoader(str(path))
    else:
        continue
    documents.extend(loader.load())

if documents:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
        retriever=retriever
    )

    user_input = st.chat_input("Írd be a kérdésed...")

    if user_input:
        with st.spinner("Gondolkodom a válaszon..."):
            result = qa_chain.run({"question": user_input, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((user_input, result))

    for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
        st.chat_message("user", avatar="👤").write(q)
        st.chat_message("assistant", avatar="🤖").write(a)
else:
    st.info("📂 Tölts fel fájlokat a 'documents/' mappába, hogy a rendszer válaszolni tudjon a kérdésekre.")
