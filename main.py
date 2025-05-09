import wave
import pyaudio
import threading
from pynput import keyboard
from speech2LLM import getTranscript

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

def on_press(key):
    global is_recording, stop_recording, frames

    try:
        if key.char == 'b':
            is_recording = not is_recording
            if is_recording:
                print("🔴 Début de l'enregistrement.")
                frames = []
            else:
                print("⏹️ Fin de l'enregistrement.")
                stop_recording = True  # Indique à la boucle d'arrêter
    except AttributeError:
        pass  # Ignore les touches spéciales

def start_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def record_audio():
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
        print("⛔ Interruption manuelle.")
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

            print(f"💾 Fichier sauvegardé : {FILENAME}")
            return FILENAME
        else:
            print("⚠️ Aucun enregistrement effectué.")
            return None

if __name__ == "__main__":
    # Démarrer le listener dans un thread
    listener_thread = threading.Thread(target=start_listener, daemon=True)
    listener_thread.start()

    # Lancer l'enregistrement
    audio_path = record_audio()
    print(getTranscript(audio_path,"medium",False))