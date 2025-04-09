from ollama import chat

file = open("codas_policy.txt", "r")
coda = file.read()

messages = [
  {
    'role': 'user',
    'content': f'You are a robot assistant with the following capabilities in the RobotActions class:  {coda} '
    'The user asks : "bring me some water from the living room, I am at Home". '
    'What action(s) do you take? Respond with only the necessary function calls in Python-like syntax.  ',
  },
]

response = chat('llama3.2:1b', messages=messages)
print(response['message']['content'])