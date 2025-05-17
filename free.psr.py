import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === Load and embed PSR from single PDF ===
@st.cache_resource(show_spinner=False)
def load_and_embed_psr():
    reader = PdfReader("PSR.pdf")
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    return vector_store

# === Load Falcon-7B model instead of Mistral ===
@st.cache_resource(show_spinner=True)
def load_local_model():
    model_id = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# === Setup QA chain ===
@st.cache_resource(show_spinner=False)
def setup_qa_chain():
    vector_store = load_and_embed_psr()
    retriever = vector_store.as_retriever()
    llm = load_local_model()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )
    return qa_chain

# === Streamlit UI ===
def main():
    col1, col2 = st.columns([1, 8])

    with col1:
        st.image("LASG.jpeg", width=400)

    with col2:
        st.markdown("<h1 style='text-align: center;'>PSR CHATBOT</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>(Built by Isaiah Tosin Ajayi)</p>", unsafe_allow_html=True)
        st.header("Lagos State Public Service Rules")
        st.write("Ask any question based on the Lagos State Public Service Rules (PSR) document.")
    
        qa_chain = setup_qa_chain()
        query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Getting answer from PSR..."):
            result = qa_chain.run(query)
            st.markdown(f"**Answer:** {result}")

if __name__ == "__main__":
    main()
