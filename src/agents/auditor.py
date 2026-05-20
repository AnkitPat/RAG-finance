import os
import json
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class AuditorAgent:
    def __init__(self, llm=None):
        # Use Gemini
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.prompt = ChatPromptTemplate.from_template("""
        You are a financial auditor. Verify the Draft Answer against the Source Context.
        Focus ONLY on numeric amounts and their scales (M vs B vs K).
        
        Draft Answer: {draft}
        Source Context: {context}
        
        Respond in JSON format:
        {{
            "is_valid": true/false,
            "discrepancies": ["list of issues found"]
        }}
        """)

    def verify(self, draft, context):
        chain = self.prompt | self.llm
        response = chain.invoke({"draft": draft, "context": context})
        
        # Strip markdown markers if present
        content = response.content.strip().replace('```json', '').replace('```', '')
        data = json.loads(content)
        return data["is_valid"], ", ".join(data["discrepancies"])
