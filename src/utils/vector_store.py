import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

class Ingestor:
    def __init__(self, collection_name="financial_docs"):
        # Using real embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GOOGLE_API_KEY.get_secret_value()
        )
        
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "]
        )

    def ingest_text(self, text, metadata):
        chunks = self.splitter.create_documents([text], metadatas=[metadata])
        self.vectorstore.add_documents(chunks)

    def search(self, query, k=3):
        return self.vectorstore.similarity_search(query, k=k)
