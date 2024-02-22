import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv

#PDF Extract api





    
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
            chunk_size = 2000,
            chunk_overlap  = 0,
            length_function = len,
        )
        if True:
            with st.spinner("Analyzing Your Document"): 
                output=ask_que(text_splitter,text)
                st.write(output)

        
        
def ask_que(text_splitter,text):
    
    
    texts = text_splitter.split_text(text)
    
    context =f"""
            ## Financial Document Processing
            This document is a financial document in PDF format, likely containing bank statements, transaction listings, or similar records. Your task is to extract specific data points related to transactions from the text. Please pay close attention to accuracy, as this is financial information.
            **Validation:**
                1. First Check if the document contains transaction data (e.g., keywords like "transaction", "amount", etc.).
                2. If no transaction data is found, return an error message indicating "Document does not contain transaction information."
                3. If transaction data is there do not show validation
                
        **Document:**

        {text}

        **Target Data:**
        
        1. **Transactions above ₹5,000:**
            - Amount (numerical value)
            - Date (formatted as YYYY-MM-DD)
            - Merchant name (text)
            - Transaction type (e.g., debit, credit, transfer)
            - Category (optional, if provided in the document)
        2. **EMI-related transactions:**
            - Amount (numerical value)
            - Date (formatted as YYYY-MM-DD)
            - Merchant name (text)
            - EMI number (if available)
            - Due date (optional, if available)
        3. **Credit card transactions:**
            - Amount (numerical value)
            - Date (formatted as YYYY-MM-DD)
            - Merchant name (text)
            - Card number (masked, last 4 digits)
            - Category (optional, if provided in the document)
        4. **Largest transaction:**
            - Amount (numerical value)
            - Date (formatted as YYYY-MM-DD)
            - Merchant name (text)
            - Transaction type (e.g., debit, credit, transfer)
            - Category (optional, if provided in the document)
        """
    chat = OpenAI(model_name='gpt-3.5-turbo-0125',temperature=0.0)
    
    
    return chat.invoke(context)
    
    
if __name__ == "__main__":
    main()
    