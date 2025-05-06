import ollama
import numpy as np

def get_embedding(text: str):
    response = ollama.embeddings(model='mxbai-embed-large', prompt=text)
    return response['embedding']

# Exemple de documents à indexer
documents = [
  "The orange juice is in the fridge",
  "The chocolate is in the cupboard",
  "Water is in the living room"
]

# Création des embeddings
embeddings = [get_embedding(doc) for doc in documents]

# Texte de la requête utilisateur (simulée)
query = "bring me some water, I am at Home"
query_embedding = get_embedding(query)

# Calcul de similarité cosinus
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Trouve le document le plus pertinent
similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
best_doc = documents[np.argmax(similarities)]

print("Requête :", query)
print("Document le plus pertinent :", best_doc)

file = open("code_as_policy.txt", "r")
coda = file.read()

messages = [
  {
    'role': 'user',
    'content': f'You are a robot assistant with the following capabilities in the RobotActions class:  {coda} '
    'The user asks : {query} '
    f'Some useful informations: "{best_doc}". '
    'What action(s) do you take? Respond with only the necessary function calls in Python-like syntax.  ',
  },
]

response = ollama.chat('llama3.2:3b', messages=messages)
print(response['message']['content'])
