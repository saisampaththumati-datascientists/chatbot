

from huggingface_hub import hf_hub_download
import ollama
from langchain_ollama import OllamaLLM

def get_llm_instance():
    llm = OllamaLLM(model="llama3.1", temperature=0)
    return llm