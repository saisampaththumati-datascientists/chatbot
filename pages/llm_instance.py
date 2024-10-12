
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Create an instance of the Ollama LLM
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_basename = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
model_path=hf_hub_download(repo_id=model_name_or_path,filename=model_basename)

def get_llm_instance():
    llm=Llama(model_path=model_path,n_batch=512,n_ctx=4000,n_gpu_layers=43,n_threads=2)
    return llm