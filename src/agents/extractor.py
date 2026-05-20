from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class ExtractorAgent:
    def __init__(self):
        # We'll leave it incomplete to test behavior, or implement as requested.
        # Task says: "Implement ExtractorAgent"
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.prompt = ChatPromptTemplate.from_template("""
        You are a financial data extractor. Use the context below to answer the query.
        Be precise with amounts.
        
        Context: {context}
        Query: {query}
        Answer:""")

    def generate_draft(self, query, context):
        chain = self.prompt | self.llm
        response = chain.invoke({"query": query, "context": context})
        return response.content
