import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


from dotenv import load_dotenv


    
load_dotenv()
    
def main():
    st.header("Made by Parag Gupta")
    st.write("Chat with PDF")
    
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()            
            
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 800,
            chunk_overlap  = 200,
            length_function = len,
        )
        user_input = st.text_input("You can search about finance from pdf")
        if(user_input):
            texts = text_splitter.split_text(text)
            chain = load_qa_chain(OpenAI(model_name='gpt-3.5-turbo'), chain_type="stuff")
            embeddings = OpenAIEmbeddings()
            document_search = FAISS.from_texts(texts, embeddings)
            docs = document_search.similarity_search(user_input)
            output =chain.run(input_documents=docs, question=user_input)
            st.write(output)
        
    
if __name__ == "__main__":
    main()
    