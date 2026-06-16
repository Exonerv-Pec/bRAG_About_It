import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "In two sentences, explain why the sky is blue."}],
)
print(response["message"]["content"])
