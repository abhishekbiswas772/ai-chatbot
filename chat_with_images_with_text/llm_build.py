from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

class LLmBuilder():
    @staticmethod
    def build_llm():
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",
            api_key=api_key
        )
        return llm
            

