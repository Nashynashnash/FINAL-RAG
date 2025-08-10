import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate  # fixed import here
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print(f"Loaded vectorstore with {db.index.ntotal} vectors")
    return db

def set_custom_prompt():
    template = """
You are an expert in understanding tweets and reading between the lines. Read all the tweets and understand and find the context and connect the dots between Text and quoted text and summarize whenever asked.
Respond to the question correctly but again never make up stories find and answer only which you completely understand. If you feel you will have to make up something to answer the user's question just say i dont know.
Be as precise as possible and make sure you will be able toi summarize efficiently and correctly.

Context:
{context}

Question:
{question}

Do remember that you should answer with relevant references you can use URL for that. Whatever you are answering shuld be mentioned with relevant reference tweets with URL.
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def load_llm():
    return ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.0,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

def main():
    st.set_page_config(page_title="Ask Your Docs", layout="wide")
    st.title("💬 Chat With Your Documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    prompt = st.chat_input("Ask a question about your text documents...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            vectors
