

from huggingface_hub import hf_hub_download
from langchain_ollama import OllamaLLM

def get_llm_instance():
    llm = OllamaLLM(model="llama3.1", temperature=0,base_url="http://0.0.0.0:11434")
    return llm