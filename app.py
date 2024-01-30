import streamlit as st
import os
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback 

from dotenv import load_dotenv
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.prompts.chat import ChatPromptTemplate
    
load_dotenv()


    
def main():
    st.header("Made by Parag Gupta")
    user_input = st.text_input("Password to acess this site")
    
    password="]5@~h18fccB("
    
    
    
    
    
    
    
    def page_pdf():
        st.write("Chat with PDF")
        
        pdf = st.file_uploader("Upload your PDF", type="pdf")
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, # it will divide the text into 800 chunk size each (800 tokens)
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)
            
            #st.write(chunks)
            
            
            ## embeddings
            
            store_name = pdf.name[:-4]
            
            
            vector_store_path = os.path.join('vector_store', store_name)
            
            
            if os.path.exists(f"{vector_store_path}.pkl"):
                with open(f"{vector_store_path}.pkl", 'rb') as f:
                    VectorStore = pickle.load(f)
                # st.write("Embeddings loaded from the Disk")
                    
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                print(VectorStore)
                with open(f"{vector_store_path}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
                
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
                
            
# GRANT ALL PRIVILEGES ON shree_sai_new.* TO 'stream'@'122.175.221.137' IDENTIFIED BY 'stream';
    def page_sql():
        st.write("Talk With SQL DB")
        username = "stream"
        password = "stream"
        host = "165.22.222.20"
        database = "shree_sai_new"
        port = "3306"  
        connection_string = f"mysql://{username}:{password}@{host}/{database}"
        db = SQLDatabase.from_uri(connection_string)
        
        st.success('Database Connection Successfully!', icon="✅")
        
        llm = OpenAI(model_name='gpt-3.5-turbo')
        agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="openai-tools",
            verbose=True
        )
        st.write(agent)
        response = agent.invoke({"input": "List the total sales per country. Which country's customers spent the most?"})
        st.write(response)
        
        # response = agent.run(question=user_input)



    if(user_input==password):
        st.write("Chat with PDF")
        
        pdf = st.file_uploader("Upload your PDF", type="pdf")
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, # it will divide the text into 800 chunk size each (800 tokens)
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)
            
            #st.write(chunks)
            
            
            ## embeddings
            
            store_name = pdf.name[:-4]
            
            
            vector_store_path = os.path.join('vector_store', store_name)
            
            
            if os.path.exists(f"{vector_store_path}.pkl"):
                with open(f"{vector_store_path}.pkl", 'rb') as f:
                    VectorStore = pickle.load(f)
                # st.write("Embeddings loaded from the Disk")
                    
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{vector_store_path}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
                
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
        # pages = {
        # "Chat With PDF ": page_pdf,
        # "Talk With SQL DB": page_sql,
        # }

        # selected_page = st.sidebar.radio("Select Page", list(pages.keys()))

        # pages[selected_page]()
        
    
    
    
      
            

    
    
    
if __name__ == "__main__":
    main()
    