

from huggingface_hub import hf_hub_download
from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_llm_instance():
    llm = OllamaLLM(
    model="llama3.1", 
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
    # llm = OllamaLLM(model="llama3.1", temperature=0,base_url="http://0.0.0.0:11434")
    return llm