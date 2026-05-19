import chromadb
from langchain_community.embeddings import FakeEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Ingestor:
    def __init__(self, collection_name="financial_docs"):
        # Using FakeEmbeddings to bypass OpenAI rate limits for testing
        self.embeddings = FakeEmbeddings(size=768)
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
