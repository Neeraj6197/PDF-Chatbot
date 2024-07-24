#importing the libraries:
import os
from io import BytesIO
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate



#loading the env variables:
load_dotenv()

#configure the genai with API KEY:
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#creating a function to read the text from the pdf:
def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text+=page.extract_text()
    return text

#defining a function to break the text into chuncks:
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

#converting the chunks into vectors and storing it:
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print(embeddings)
    vectore_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vectore_store.save_local("faiss_index")


#defining a conversational chain:
def get_conversational_chain():
    prompt_template = """"
    From the document answer the following question based on the context.
    Context:{context}
    Question:{question}
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context','question'])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

#creating a function to generate and display the chat:
def user_input(user_question):
    #generating the embeddings:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    #loading the created embeddings
    new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)

    #searching for similar vectors:
    docs = new_db.similarity_search(user_question)

    #loading the chain:
    chain = get_conversational_chain()
    
    #getting the response:
    response = chain(
        {"input_documents":docs,
         "question":user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ",response["output_text"])


#creating the main fucntion with UI:
def main():
    st.set_page_config(page_title="PDF ChatBot")
    st.header("CHAT with your PDF")

    user_question = st.text_input("Ask a question")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDFs here",type=["pdf"], accept_multiple_files=False)
        if st.button("Submit"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done.")


if __name__ == "__main__":
    main()