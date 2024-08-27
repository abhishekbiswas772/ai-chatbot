from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


class LLMHandler:
    @staticmethod
    def load_llm_model():
        load_dotenv()
        open_api_key = os.getenv("OPEN_API_KEY")
        llm = ChatOpenAI(
            model = "gpt-4o-mini",
            api_key=open_api_key,
            temperature=0.3
        )
        return llm
    
        


    
