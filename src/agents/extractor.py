import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class ExtractorAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.prompt = ChatPromptTemplate.from_template("""
        You are a financial data extractor. Use the context below to answer the query.
        Be precise with amounts.
        
        Context: {context}
        Query: {query}
        Answer:""")

    def generate_draft(self, query, context):
        print(f"Extractor: Generating draft for query: {query[:50]}...")
        import time
        start_time = time.time()
        
        chain = self.prompt | self.llm
        response = chain.invoke({"query": query, "context": context})
        
        end_time = time.time()
        print(f"Extractor: Draft generated in {end_time - start_time:.2f} seconds.")
        return response.content
