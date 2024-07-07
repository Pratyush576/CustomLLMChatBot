import streamlit as st
import os
import pickle
import langchain
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import UnstructuredURLLoader

from dotenv import load_dotenv

#"""
#Set's the url to be used for content generation
#"""
def read_custom_urls(st):
    url1 = st.text_input('URL1') # Take input from the user
    url2 = st.text_input('URL2') # Take input from the user
    url3 = st.text_input('URL3') # Take input from the user

    # Default url's
    url4 = "https://finance.yahoo.com/news/youd-invested-1-000-nvidia-155700457.html"
    url5 = "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-first-quarter-fiscal-2025"
    
    urls = [url1, url2, url3, url4, url5]
    return urls

#"""
# Initialize openai LLM
#"""
def initialize_openai_llm():
    llm = OpenAI(temperature=0.9, max_tokens=500)
    return llm


#"""
# Create FAISS vector index
#"""
def create_faiss_vector_index(docs):
    embeddings = OpenAIEmbeddings()

    # Pass the documents and embeddings inorder to create FAISS vector index
    vectorindex_openai = FAISS.from_documents(docs, embeddings)
    return vectorindex_openai

#"""
# generate the reponse using openai llm
#"""
def generate_response(st, question, chain):
    result = chain({"question": question}, return_only_outputs=True)
    st.header("Question")
    st.markdown(question)
    st.header("Result")
    st.markdown(''':tulip: :red[Result] ''' + result['answer'])
    st.markdown(''':rose: :red[Source] ''' + result['sources'])


st.title("My First Streamlit")
load_dotenv()  # take environment variables from .env (especially openai api key)
urls = read_custom_urls(st) # sets the web URL's for content generation

# load the urls
loader = UnstructuredURLLoader(urls)
data = loader.load()


# As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(data)

# Create the embeddings of the chunks using openAIEmbeddings
# Initialise LLM with required params
llm = initialize_openai_llm()

# Pass the documents and embeddings inorder to create FAISS vector index
vectorindex_openai = create_faiss_vector_index(docs)
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex_openai.as_retriever())

# Encode the search text using same encoder and normalize the output vector
langchain.debug=True
question = st.text_input('Ask Question')

if st.button('Show Result') and question.strip():
    generate_response(st, question, chain)


