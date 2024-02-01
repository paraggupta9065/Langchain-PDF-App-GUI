import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback

from dotenv import load_dotenv


    
load_dotenv()
    
def main():
    st.header("Made by Parag Gupta")
    st.write("Chat with PDF")
    
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    #st.write(pdf)
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()            
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, # it will divide the text into 800 chunk size each (800 tokens)
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        #st.write(chunks)
        
        
        ## embeddings
        
        
        
        
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            
            # st.write("Embeddings Computation Completed ")
            
        
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file")
        #st.write(query)
        
        if query:
            
            docs = VectorStore.similarity_search(query=query, k=3) # k return the most relevent information
            
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type='stuff')
            with get_openai_callback() as cb:
                
                response = chain.run(input_documents=docs, question=query)
            st.write(response)
            
            
            

    
    
    
if __name__ == "__main__":
    main()
    