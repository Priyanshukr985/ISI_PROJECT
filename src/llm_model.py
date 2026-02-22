from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

class LLM_Loader:
    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL")
        self.api_key = os.getenv("GROQ_API_KEY")

    def load(self):
        """Load Hugging Face embedding model wrapper for LangChain."""
        return ChatGroq(
            api_key=self.api_key,
            model=self.model_name
        )