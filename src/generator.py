import os
import re
from llama_cpp import Llama
from config import *

class LLMGenerator:
    """A class to interact with the Llama model for text generation."""

    def __init__(self, model_path: str = MODEL_PATH, n_ctx: int = N_CTX, verbose: bool = False) -> None:
        """Initialize the Llama model.
        Args:
            model_path (str): Path to the Llama model file.
            n_ctx (int): Context size for the model.
            verbose (bool): Whether to print verbose output.
        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if os.path.exists(model_path):
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                verbose=verbose,
                stop=STOP,
                top_k=TOP_K,
                top_p=TOP_P,
                repeat_penalty=REPEPETITION_PENALTY,
                n_threads=os.cpu_count()
            )
            print("LLM Loaded Successfully")
        else:
            raise FileNotFoundError("Model file not found! Download it to 'models/'")

    def output_parser(self, raw_response: str) -> str:
        """Cleans up LLM output by removing unwanted characters and expression patterns.
        Args:
            raw_response (str): The raw response from the LLM.
        Returns:
            str: A cleaned version of the response.
        """
        if not raw_response:
            return ""

        cleaned_response = re.sub(r'[\n\t\r]', ' ', raw_response).strip()

        patterns_to_remove = [
            r'^\s*One phrase answer:\s*',
            r'^\s*answer:\s*',
            r'^\s*assistant:\s*',
            r'^\s*AI:\s*',
            r'^\s*Response:\s*',
            r'^\s*User:\s*',
            r'^\s*Chatbot:\s*',
            r'^\s*System:\s*',
            r'^\s*A:\s*',
        ]

        for pattern in patterns_to_remove:
            cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.IGNORECASE).strip()

        return cleaned_response
    
    def count_tokens(self, prompt: str) -> int:
        """Returns the number of tokens in the input string."""
        return len(self.llm.tokenize(prompt.encode("utf-8")))

    def llmgenerate(self, prompt: str, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE) -> str:
        """Queries the LLM model and returns the cleaned response.
        Args:
            prompt (str): The input prompt for the LLM.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature for generation.
        Returns:
            str: The cleaned response from the LLM.
        """
        raw_response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )["choices"][0]["text"]

        return self.output_parser(raw_response)