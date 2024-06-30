import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def get_pdf_tex(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdfReader = PyPDF2.PdfReader(pdf)
        for page in pdfReader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = faiss.FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context,make sure to provide all the details,if the answer is not in the provided context just say, "Answer is not available in the context",don't provide the wrong answer
    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = faiss.FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({
        "input_documents": docs,
        "question": user_question
    }, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")
    user_question = st.text_input("Ask a question from the PDF files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("PDF Files")
        pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing PDF files"):
                pdf_text = get_pdf_tex(pdf_files)
                chunks = get_text_chunks(pdf_text)
                get_vector_store(chunks)
                st.success("PDF files processed successfully")


if __name__ == '__main__':
    main()