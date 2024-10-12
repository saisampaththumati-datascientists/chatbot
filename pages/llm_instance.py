

from huggingface_hub import hf_hub_download
import ollama
from langchain_ollama import OllamaLLM

local_model = "mistral"

llm = OllamaLLM(model="llama3.1", temperature=0)
# Create an instance of the Ollama LLM
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_basename = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
model_path=hf_hub_download(repo_id=model_name_or_path,filename=model_basename)

def get_llm_instance():
    llm = OllamaLLM(model="llama3.1", temperature=0)
    return llm