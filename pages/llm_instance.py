from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download

# Create an instance of the Ollama LLM
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_basename = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
model_path=hf_hub_download(repo_id=model_name_or_path,filename=model_basename)

def get_llm_instance():
    llm=LlamaCpp(model_path=model_path,n_batch=512,n_ctx=6000,n_gpu_layers=43,n_threads=2)
    return llm