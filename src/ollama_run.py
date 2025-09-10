"""
Ollama Model Temperature Testing Script

What is Ollama?
- Ollama is an open-source platform for running large language models locally
- It provides a simple command-line interface to download and run pre-trained models
- Unlike cloud-based APIs, Ollama runs models entirely on your local machine
- This enables privacy, offline usage, and comparison with local custom models
- Models are optimized for local hardware and can run on CPUs or GPUs
- Popular for running models like Llama, Mistral, and custom fine-tuned models

This script demonstrates the effect of temperature settings on text generation using Ollama models.
It tests a range of temperature values (0.0 to 2.0) on a simple prompt to show how temperature
affects the creativity and determinism of generated responses.

Purpose:
- Compare different temperature settings on the same prompt
- Demonstrate temperature's impact on model creativity vs consistency
- Provide baseline for understanding temperature parameter in text generation
- Test Ollama API integration and model availability

Temperature Scale:
- 0.0: Completely deterministic (always same output)
- 0.3: Very low creativity, highly consistent
- 0.7: Balanced creativity and coherence
- 1.0: Moderate creativity (Ollama default)
- 1.3: High creativity with some coherence
- 1.5: Very high creativity
- 2.0: Maximum creativity, potentially incoherent

Test Configuration:
- Model: JetBrains/Mellum-4b-base (lightweight instruction-tuned model)
- Prompt: Simple number repetition task to isolate temperature effects
- Max Tokens: 10 (limited to focus on temperature impact)
- Temperatures: 7 different values for comprehensive comparison

Usage:
    python ollama_run.py

Prerequisites:
    - Ollama must be installed and running
    - Model must be downloaded: ollama pull JetBrains/Mellum-4b-base
    - ollama_python package must be installed

Expected Output:
    For each temperature setting, the script will output:
    - Temperature value being tested
    - Generated response from the model
    - Error messages if model is unavailable

Learning Outcomes:
- Understanding how temperature affects generation diversity
- Recognizing the trade-off between creativity and consistency
- Identifying appropriate temperature ranges for different use cases
- Comparing local model performance vs custom transformer models

Dependencies:
    - ollama_python: Python client for Ollama API
    - Ollama: Local LLM server (external dependency)
"""

from ollama_python.endpoints import GenerateAPI

model = "JetBrains/Mellum-4b-base"
temperatures = [0.0, 0.3, 0.7, 1.0, 1.3, 1.5, 2.0]

prompt = """
Question:
Repeat this number back to me with no other text at all - 123456
Answer:
"""

# Initialize the API client with the model
api = GenerateAPI(model=model)

for temp in temperatures:
    print(f"\n--- Running with temperature={temp} ---")
    try:
        response = api.generate(
            prompt=prompt, options={"temperature": temp, "num_predict": 10}
        )
        print(response.response)
    except Exception as e:
        print(f"Error: {e}")
        print(
            "Make sure the model is available. Try: ollama pull JetBrains/Mellum-4b-base"
        )
