from faster_whisper import WhisperModel
import time

model_size = "large-v3"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8

model = WhisperModel(model_size, device="cpu", compute_type="int8")
start = time.time()
segments, info = model.transcribe("samples/ines_test.m4a", beam_size=5)
end = time.time()
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
print("Transcription took %.2fs" % (end - start))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))