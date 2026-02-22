from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

import os

load_dotenv()

class PDFLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def load(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
        loader = PyPDFLoader(self.pdf_path)

        return loader.load()
    

class TextChunker:
    def __init__(self, chunk_size = 1000, chunk_overlap = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap =   self.chunk_overlap
        )

        return splitter.split_documents(docs)
    
class HFEmbedding:
    def __init__(self, model_name = "BAAI/bge-large-en-v1.5", device = "cpu"):
        self.model_name = os.getenv("EMBEDDING_MODEL")
        self.device = device

    def load(self):
        return HuggingFaceEmbeddings(
            model_name = self.model_name,
            model_kwargs = {"device": self.device},
            encode_kwargs = {"normalize_embeddings": True}
        )

class FAISSStore:
    def __init__(self, embedding_model, index_path = "faiss_index"):
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.index_path = index_path

    def build(self , documents):
        self.vectorstore = FAISS.from_documents(
            documents = documents,
            embedding=self.embedding_model
        )

        return self.vectorstore
    
    def save(self):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not built. Cannot save")
        
        os.makedirs(self.index_path, exist_ok=True)
        self.vectorstore.save_local(self.index_path)
        print(f"✔ FAISS index saved at: {self.index_path}")

    def load(self):
          if not os.path.exists(self.index_path):
              raise FileNotFoundError(f"No FAISS index found in {self.index_path}")
          
          self.vectorstore = FAISS.load_local(
              folder_path=self.index_path,
              embeddings=self.embedding_model,
              allow_dangerous_deserialization=True
          )

          print(f"✔ FAISS index loaded from: {self.index_path}")
          return self.vectorstore
    
    def get_retriever(self, vectordb):
        """Return retriever object."""
        if vectordb is None:
            raise ValueError("Vectorstore not initialized")
        return vectordb.as_retriever()
    

        

