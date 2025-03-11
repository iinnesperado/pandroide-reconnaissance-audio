from faster_whisper import WhisperModel
import time
import os
import glob

model_size = "large-v3"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8

path = "/Users/ines/androide/p-androide/samples"

model = WhisperModel(model_size, device="cpu", compute_type="int8")
timeFile = open("data/exec_time.txt", "w")
scriptFile = open("data/fw_scripts.txt", "w")
i = 1

for name in glob.glob('samples/*.m4a'):
    start = time.time()
    segments, info = model.transcribe(name, beam_size=5)
    end = time.time()
    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    #print("Transcription took %.2fs" % (end - start))

    timeFile.write(f"{name}\t")
    timeFile.write("%.2f\n" % (end - start))
    
    scriptFile.write("\n(%d) "% i)
    scriptFile.write("Transcription of %s" % (name))
    for segment in segments:
        scriptFile.write("%s" % (segment.text))
    
    i += 1

timeFile.close()
scriptFile.close()