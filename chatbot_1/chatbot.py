from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyMuPDFLoader,PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from  langchain_core.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

load_dotenv()

#api_key=os.getenv("OPENAI_API_KEY")
#pinecone_key=os.getenv("PINECONE_API_KEY")

api_key = st.secrets["OPENAI_API_KEY"]
pinecone_key=st.secrets["PINECONE_API_KEY"]

os.environ["OPENAI_API_KEY"]=api_key
#os.environ["PINECONE_API_KEY"]=pinecone_key

#openai.api_key = st.secrets["OPENAI_API_KEY"]


def load_pdf(directory):
    loader=PyPDFDirectoryLoader(directory,glob='*.pdf')
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
    documents=text_splitter.split_documents(data)
    return documents



## Connect to Pinecone server
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_key)

index_name = "chatbot2"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 

embeddings=OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)


extracted_data=load_pdf(r'C:\Users\Hp\Desktop\cleave\data\Medical_book (1).pdf')

vectorstore_from_docs = PineconeVectorStore.from_documents(
        extracted_data,
        index_name=index_name,
        embedding=embeddings
)

# Step 1: Create a custom callback handler for streaming
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.output = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)  # Print tokens in real-time
        self.output += token  # Collect tokens if needed

# Step 2: Initialize streaming-compatible LLM and callback
stream_handler = StreamingCallbackHandler()

llm = OpenAI(
    streaming=True,  # Enable streaming
    callbacks=[stream_handler]  # Attach the callback handler
)
embeddings = OpenAIEmbeddings()

prompt_template='''
You are a helpful assistant.
Use the following pieces of information to answer the users questions.
if you dont know the answer , just say that you dont know . dont try to 
make up the answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful Answer :
'''

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_from_docs.as_retriever(),
    return_source_documents=True
)

def retriever(qa,query):
    return qa({'query':query})['result']

#print(retriever(qa,{'query':'Hello, i need some h