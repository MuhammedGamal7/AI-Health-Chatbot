#LAMAAAAAA
import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings  # Required base class

# Load .env or Streamlit secrets
load_dotenv()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Missing API key. Please add it to .env or Streamlit Secrets.")
    st.stop()

# Define embedding wrapper that complies with LangChain
class EmbeddingsWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Cache model loading
@st.cache_resource
def load_model():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # No .to(device) for Streamlit Cloud
    return model

# Load model and wrapper
sentence_transformer_model = load_model()
embeddings = EmbeddingsWrapper(sentence_transformer_model)

# Streamlit UI
st.title("Medical Chatbot With Chat History")
st.write("Chat Now!")

# Initialize Groq LLM
try:
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    
except Exception as e:
    st.error(f"Failed to initialize LLM: {e}")
    st.stop()

# Session state
session_id = st.text_input("Session ID", value="default_session")
if 'store' not in st.session_state:
    st.session_state.store = {}

# PDF source
predefined_pdfs = ["Health Montoring Box (CHATBOT).pdf"]

# Load and split PDFs
@st.cache_data
def load_and_process_pdfs(pdf_paths):
    docs = []
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)
            docs += loader.load()
        except Exception as e:
            st.error(f"Error processing {path}: {e}")
    return docs

if predefined_pdfs:
    documents = load_and_process_pdfs(predefined_pdfs)
    st.success(f"Loaded {len(documents)} pages from {len(predefined_pdfs)} PDF(s).")

    # Vector store generation
    @st.cache_resource
    def generate_embeddings(_documents):
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        chunks = splitter.split_documents(_documents)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore

    try:
        vectorstore = generate_embeddings(documents)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Error building vectorstore: {e}")
        st.stop()

    # Prompt setup
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, reformulate it as a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are a medical assistant. Use the context to answer concisely. "
        "If it's a dangerous medical situation, advise seeing a doctor."
        "\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Session chat history
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    rag_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Input and response
    user_input = st.text_input("Your question:", key="user_input", on_change=lambda: st.session_state.update({"submitted": True}))
    submit_pressed = st.button("Submit")

    if submit_pressed or st.session_state.get("submitted"):
        st.session_state["submitted"] = False
        if user_input:
            with st.spinner("Generating response..."):
                try:
                    response = rag_with_history.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )
                    final_answer = response["answer"].split("</think>")[-1].strip() if "</think>" in response["answer"] else response["answer"]
                    st.markdown(f"Answer: {final_answer}")
                    with st.expander("Chat History"):
                        st.write(get_session_history(session_id).messages)
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.button("Clear Chat History"):
        st.session_state.store[session_id] = ChatMessageHistory()
        st.success("Chat history cleared.")


