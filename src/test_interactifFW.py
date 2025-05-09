import queue
import re
import threading
import time
from typing import Dict, List
import keyboard
from speech2LLM import getTranscript
import wave

import numpy as np
import pyaudio
from faster_whisper import WhisperModel

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
FILENAME = "recorded_audio.wav"
"""
# Yeah I could do this config with argparse, but I won't...

# Audio settings
STEP_IN_SEC: int = 1    # We'll increase the processable audio data by this
LENGHT_IN_SEC: int = 6    # We'll process this amount of audio data together maximum
NB_CHANNELS = 1
RATE = 16000
CHUNK = RATE

# Whisper settings
WHISPER_LANGUAGE = "en"
WHISPER_THREADS = 4

# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 80

# This queue holds all the 1-second audio chunks
audio_queue = queue.Queue()

# This queue holds all the chunks that will be processed together
# If the chunk is filled to the max, it will be emptied
length_queue = queue.Queue(maxsize=LENGHT_IN_SEC)

# Whisper model
whisper = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=WHISPER_THREADS, download_root="./models")


def producer_thread():
    if keyboard.is_pressed("q"):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=NB_CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,    # 1 second of audio
        )

        print("-" * 80)
        print("Microphone initialized, recording started...")
        print("-" * 80)
        print("TRANSCRIPTION")
        print("-" * 80)

        while True:
            audio_data = b""
            for _ in range(STEP_IN_SEC):
                chunk = stream.read(RATE)    # Read 1 second of audio data
                audio_data += chunk

            audio_queue.put(audio_data)    # Put the 5-second audio data into the queue


# Thread which gets items from the queue and prints its length
def consumer_thread(stats):
    while True:
        if length_queue.qsize() >= LENGHT_IN_SEC:
            with length_queue.mutex:
                length_queue.queue.clear()
                print()

        audio_data = audio_queue.get()
        transcription_start_time = time.time()
        length_queue.put(audio_data)

        # Concatenate audio data in the lenght_queue
        audio_data_to_process = b""
        for i in range(length_queue.qsize()):
            # We index it so it won't get removed
            audio_data_to_process += length_queue.queue[i]

        # convert the bytes data toa  numpy array
        audio_data_array: np.ndarray = np.frombuffer(audio_data_to_process, np.int16).astype(np.float32) / 255.0
        # audio_data_array = np.expand_dims(audio_data_array, axis=0)

        segments, _ = whisper.transcribe(audio_data_array,
                                         language=WHISPER_LANGUAGE,
                                         beam_size=5,
                                         vad_filter=True,
                                         vad_parameters=dict(min_silence_duration_ms=1000))
        segments = [s.text for s in segments]

        transcription_end_time = time.time()

        transcription = " ".join(segments)
        # remove anything from the text which is between () or [] --> these are non-verbal background noises/music/etc.
        transcription = re.sub(r"\[.*\]", "", transcription)
        transcription = re.sub(r"\(.*\)", "", transcription)
        # We do this for the more clean visualization (when the next transcription we print would be shorter then the one we printed)
        transcription = transcription.ljust(MAX_SENTENCE_CHARACTERS, " ")

        transcription_postprocessing_end_time = time.time()

        print(transcription, end='\r', flush=True)

        audio_queue.task_done()

        overall_elapsed_time = transcription_postprocessing_end_time - transcription_start_time
        transcription_elapsed_time = transcription_end_time - transcription_start_time
        postprocessing_elapsed_time = transcription_postprocessing_end_time - transcription_end_time
        stats["overall"].append(overall_elapsed_time)
        stats["transcription"].append(transcription_elapsed_time)
        stats["postprocessing"].append(postprocessing_elapsed_time)


if __name__ == "__main__":
    stats: Dict[str, List[float]] = {"overall": [], "transcription": [], "postprocessing": []}



    producer = threading.Thread(target=producer_thread)
    producer.start()

    consumer = threading.Thread(target=consumer_thread, args=(stats,))
    consumer.start()

    try:
        producer.join()
        consumer.join()
    except KeyboardInterrupt:
        print("Exiting...")
        # print out the statistics
        print("Number of processed chunks: ", len(stats["overall"]))
        print(f"Overall time: avg: {np.mean(stats['overall']):.4f}s, std: {np.std(stats['overall']):.4f}s")
        print(
            f"Transcription time: avg: {np.mean(stats['transcription']):.4f}s, std: {np.std(stats['transcription']):.4f}s"
        )
        print(
            f"Postprocessing time: avg: {np.mean(stats['postprocessing']):.4f}s, std: {np.std(stats['postprocessing']):.4f}s"
        )
        # We need to add the step_in_sec to the latency as we need to wait for that chunk of audio
        print(f"The average latency is {np.mean(stats['overall'])+STEP_IN_SEC:.4f}s")
"""

# version avec enresgistrement audio
def record_audio():
    is_recording = False
    frames = []

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Press 'b' to start/stop the record")

    while True:
        if is_recording:
            data = stream.read(CHUNK)
            frames.append(data)

        if keyboard.is_pressed('b'):
            is_recording = not is_recording
            if is_recording:
                print(" Start of the recording")
                frames = []
            else:
                print("End of the recording")
                break
            while keyboard.is_pressed('b'):  # avoid multiple toggles
                pass

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save WAV file
    wf = wave.open(FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Fichier sauvegardé : {FILENAME}")
    return FILENAME


"""
# Config audio
FORMAT = pyaudio.paInt16

is_recording = False

# Choose the whisper model
model = WhisperModel("large-v3", device="cpu", compute_type="int8")

def listen_keyboard():
    global is_recording
    print("Press 'b' to start/stop the transcription (Ctrl+C to quit).")
    while True:
        if keyboard.is_pressed('b'):
            is_recording = not is_recording
            if is_recording:
                print("Start of the transcription")
            else:
                print("End of the transcription")

def audio_stream():
    global is_recording
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    buffer = []

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if is_recording:
                buffer.append(data)

                # If we reach 1 second of audio
                if len(buffer) * CHUNK >= RATE:
                    audio_bytes = b''.join(buffer)
                    buffer = []

                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 255.0

                    # Transcription
                    segments, _ = model.transcribe(audio_array,
                                                   beam_size=5,
                                                   vad_filter=True,
                                                   vad_parameters=dict(min_silence_duration_ms=500))

                    transcription = " ".join([s.text for s in segments])
                    # remove anything from the text which is between () or [] --> these are non-verbal background noises/music/etc.
                    transcription = re.sub(r"\[.*\]", "", transcription)
                    transcription = re.sub(r"\(.*\)", "", transcription)

                    if transcription.strip():
                        print(f"Transcription {transcription}")
            else:
                buffer = []
            time.sleep(0.01)  # Pause to avoid CPU saturation

    except KeyboardInterrupt:
        print("\nInterruption by the user")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
"""

if __name__ == "__main__":
    #threading.Thread(target=listen_keyboard, daemon=True).start()
    #audio_stream()
    audio = record_audio()
    print(getTranscript(audio,False))
