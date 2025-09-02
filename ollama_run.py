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
