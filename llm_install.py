from llama_cpp import Llama

# Load the LLM model from Hugging Face (or locally if already downloaded)
llm = Llama.from_pretrained(
    repo_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',  # Repository containing the model
    filename='mistral-7b-instruct-v0.2.Q4_K_M.gguf',   # Model file to download/use
    local_dir='models',  # Directory where the model should be stored
    verbose=True  # Enables logging to show download progress and loading status
)

# The model will be automatically downloaded if not present in the 'models' folder.
# If it's already downloaded, it will load the model from local storage.