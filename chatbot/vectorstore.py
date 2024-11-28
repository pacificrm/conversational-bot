import pinecone
from langchain.vectorstores import Pinecone
from chatbot.config import PINECONE_API_KEY, PINECONE_API_ENV

def initialize_pinecone(index_name, embeddings, documents):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    vectorstore = Pinecone.from_documents(documents, embeddings, index_name=index_name)
    return vectorstore
