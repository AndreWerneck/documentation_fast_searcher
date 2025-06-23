import pytest
from src.generator import LLMGenerator

@pytest.fixture(scope="module")
def llm():
    return LLMGenerator()

def test_generate_basic_answer(llm):
    prompt = "Explain what a SageMaker pipeline is in simple terms."
    output = llm.llmgenerate(prompt)
    assert isinstance(output, str)
    assert len(output.strip()) > 10  # Ensure non-empty output