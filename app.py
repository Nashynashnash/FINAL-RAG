import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def set_custom_prompt():
    template = """
Use the following context to answer the user's question.
If the context provides indirect clues, use reasoning to infer the answer.
If you are unsure or the answer is not in the context, just say "I don't know."

Context:
{context}

Question:
{question}

Answer:
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
            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt()}
            )
            response = qa_chain.invoke({"query": prompt})
            result = response["result"]
            sources = response["source_documents"]
            source_texts = "\n".join(f"• {doc.metadata.get('source', 'Unknown')}" for doc in sources)

            final_output = result + "\n\n📄 **Sources:**\n" + source_texts

            st.chat_message("assistant").markdown(final_output)
            st.session_state.messages.append({"role": "assistant", "content": final_output})

        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

    
