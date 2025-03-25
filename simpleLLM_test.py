from ollama import chat

messages = [
  {
    'role': 'user',
    'content': 'What day are we today ?',
  },
]

response = chat('llama3.2:1b', messages=messages)
print(response['message']['content'])