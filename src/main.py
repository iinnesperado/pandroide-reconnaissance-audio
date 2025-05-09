import wave
import pyaudio
import threading
from pynput import keyboard
from whisper_processor import getTranscript
import ollama
import numpy as np

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
FILENAME = "recorded_audio.wav"

# Shared state
is_recording = False
stop_recording = False
frames = []

def reset_audio_state():
    global is_recording, stop_recording, frames
    is_recording = False
    stop_recording = False
    frames = []


def on_press(key):
    ''' Gère les événements de pression de touche. '''
    global is_recording, stop_recording, frames

    try:
        if key.char == 'b':
            is_recording = not is_recording
            if is_recording:
                print("Début de l'enregistrement.")
                frames = []
            else:
                print("Fin de l'enregistrement.")
                stop_recording = True  # Indique à la boucle d'arrêter
    except AttributeError:
        pass  # Ignore les touches spéciales

def start_listener():
    ''' Démarre un listener pour les événements du clavier. '''
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def record_audio():
    ''' Enregistre l'audio à partir du microphone et le sauvegarde dans un fichier WAV. '''
    global is_recording, stop_recording, frames

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Appuie sur 'b' pour démarrer/arrêter l'enregistrement (Ctrl+C pour quitter)")

    try:
        while not stop_recording:
            if is_recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
    except KeyboardInterrupt:
        print("Interruption manuelle.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

        if frames:
            wf = wave.open(FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            print(f"Fichier sauvegardé : {FILENAME}")
            return FILENAME
        else:
            print("Aucun enregistrement effectué.")
            return None

def get_embedding(text: str):
    response = ollama.embeddings(model='mxbai-embed-large', prompt=text)
    return response['embedding']

# Calcul de similarité cosinus
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_best_doc(query, documents):
    ''' Trouve le document le plus pertinent par rapport à la requête. '''
    embeddings = [get_embedding(doc) for doc in docs]
    query_embedding = get_embedding(query)
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    best_doc = documents[np.argmax(similarities)]
    return best_doc



if __name__ == "__main__":
    # Exemple de documents à indexer
    docs = [
    "The orange juice is in the fridge",
    "The chocolate is in the cupboard"
    ]
    # Démarrer le listener dans un thread
    listener_thread = threading.Thread(target=start_listener, daemon=True)
    listener_thread.start()

    # Lancer l'enregistrement
    reset_audio_state()
    audio_path = record_audio()
    
    query = getTranscript(audio_path,"large-v3",False)
    print("Requête :", query)

    best_doc = get_best_doc(query, docs)
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

    reset_audio_state()
    audio_path2 = record_audio()
    query2 = getTranscript(audio_path2,"large-v3",False)
    print("Requête :", query2)

    docs.append(query2)
    best_doc2 = get_best_doc(query2, docs)

    print("Document le plus pertinent :", best_doc2)

    messages2 = [
    {
        'role': 'user',
        'content': f'You are a robot assistant with the following capabilities in the RobotActions class:  {coda} '
        'The user asks : {query} '
        f'Some useful informations: "{best_doc2}". '
        'What action(s) do you take? Respond with only the necessary function calls in Python-like syntax.  ',
    },
    ]

    response = ollama.chat('llama3.2:3b', messages=messages2)
    print(response['message']['content'])