

from huggingface_hub import hf_hub_download
from langchain_ollama import OllamaLLM
import requests

try:
    response = requests.get("http://127.0.0.1:11434")
    print(response.text)
except requests.ConnectionError as e:
    print(f"Connection error: {e}")

def get_llm_instance():
    llm = OllamaLLM(model="llama3.1", temperature=0, base_url="http://127.0.0.1:11434")

    return llm