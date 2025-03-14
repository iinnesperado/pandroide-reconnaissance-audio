from ollama import chat

messages = [
  {
    'role': 'user',
    'content': 'How should I call you?',
  },
]

response = chat('llama3.2:1b', messages=messages)
print(response['message']['content'])