from ollama import chat

messages = [
  {
    'role': 'user',
    'content': 'Calcule le double de 10',
  },
]

response = chat('llama3.2:1b', messages=messages)
print(response['message']['content'])