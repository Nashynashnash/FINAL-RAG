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
    embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def set_custom_prompt():
    template = """
You are an intelligent AI assistant trained to deeply analyze documents and answer questions with maximum accuracy and reasoning.

Use ONLY the following STORY as your context. Never hallucinate or make assumptions beyond this context unless instructed. 
Base your conclusions strictly on the content provided, citing exact lines or sections to support your answers.

Key Rules to Follow:
1. If the answer is clearly stated, extract it concisely and back it with the exact line(s).
2. If the answer is indirect, reason step-by-step using clues from the text and cite the logic chain with text snippets.
3. If the answer cannot be determined from the STORY, respond with: "I don't know based on the provided context."
4. Never make up information, guess names, or add personality or pleasantries.
5. If asked to summarize, create a logically organized and complete summary with clear references to sections or themes.

You are expected to:
- Be precise, not verbose.
- Show your reasoning when inferring.
- Always cite where your answer comes from in the STORY.
- Avoid fluff and praise. Focus only on accuracy and traceability.

STORY:
{context}

QUESTION:
{question}

Now give a clear, factual, citation-backed answer:
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

    
