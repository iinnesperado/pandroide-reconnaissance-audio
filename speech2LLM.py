from faster_whisper import WhisperModel
from ollama import chat
import time
import collections
import re
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# All mentions of 'data' refer to : 
#     - execution time
#     - accuracy score
# transcript and transcription are interchangeable

# Audio processing stuff #

def getTranscript(audioPath, model_size, record):
    '''
    Fait la transcription du fichier audio donnée
    
    :param audioPath : str - path to audio file (m4a or mp3)
    :params record : bool - if True it writes the data into a file, else just prints it
    :return transcript : str - transcription of the audio
    '''

    # model_size = "large-v3"

    # Run on GPU with FP16
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    
    # or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    start = time.time()
    segments, info = model.transcribe(audioPath, beam_size=5)
    end1 = time.time()
    transcribe_t = end1 - start

    transcript = ""
    for segment in segments:
        # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        transcript += segment.text

    end2 = time.time()
    total_t = end2 - start
    if record:
        directory = "data/" + model_size
        if not os.path.exists(directory):
            os.makedirs(directory)
        saveData(audioPath, model_size + "/exec_time", total_t)
        # to plot a comparison of exec time of func transcribe and the time with saving text
        file = open("data/" + model_size + "/comparison_exec_time.txt", "a")
        file.write("%.2f\t%.2f\n" % (transcribe_t, total_t))
        file.close()
    else :
        print("Transcription of audio '%s' took : %.2fs" % (getFileName(audioPath),total_t))
    
    transcript_dir = "transcriptions/" + model_size
    if not os.path.exists(transcript_dir):
        os.makedirs(transcript_dir)
    saveTranscript(transcript, transcript_dir + "/fw_"+ getFileName(audioPath) + ".txt")

    return transcript

def saveTranscript(transcript, filePath):
    '''
    Creates a file and saves the transcriptions of the audioPath done by faster-whisper
    as well as record the execution time into ecxec_time.txt file
    
    :param transcript : str - content (stranscript) to save into a file
    :param filePath : str - file path to the file where the transcriptions will be saved
    :return void
    '''

    file = open(filePath, "w")
    file.write(transcript)
    file.close()

def saveData(audioPath, dataType, data):
    '''
    Saves the data (execution time or score) into a file in the directory 'data'

    :params audioPath : str - reference to know to wich audio file the data is from
    :params dataType : str - name of the file where the data is going to be saved
    :params data : float - the data to save
    :return void
    '''
    try:
        file = open("data/" + dataType + ".txt", "a")
    except:
        file = open("data/" + dataType + ".txt", "w")
    file.write(getFileName(audioPath) + "\t%.2f\n" % (data))
    file.close()

def getScore(audioPath, og_file, model_size, record):
    '''
    Gives accuracy score to the transcription done by faster-wshiper out of 100.
    Score = num of words missing in fw_transcript (compared to og_text)/ total words in og_text
    
    :params audioPath : str - path to audio file (m4a or mp3)
    :param  og_file : str - file path to transcription
    :params record : bool - if True it writes the data into a file, else just prints it
    :return score : float
    '''
    # Transcription faite par faster whisper
    fw_file = "transcriptions/" + model_size + "/fw_"+getFileName(audioPath)+".txt"
    fwFile = open(fw_file, "r")
    fw_read = fwFile.read()
    fwFile.close()

    # Transcription faites par nous utilisée comme référence
    if og_file==None:
        og_file = "transcriptions/original/og_"+getFileName(audioPath)+".txt"
    ogFile = open(og_file, "r")
    og_read = ogFile.read()
    ogFile.close()

    # On transforme le texte en liste de mots
    fw_text = list(filter(None, re.split(r"[,.?!\s\t\n]\s*", fw_read)))
    og_text = list(filter(None, re.split(r"[,.?!\s\t\n]\s*", og_read)))

    total_mots = len(og_text)

    # missedWords donne les mots qui ne sont pas dans fw_text par rapport à og_text
    missedWords = set(og_text) - set(fw_text)

    # Transforme la liste de mots og_text en dictionaire de fréquences de mots
    # pour savoir le nombre total de mots qui manquent (cas où un mots serait plusieurs fois
    # mal compris)
    og_dict = collections.Counter(og_text)
    cpt = 0 
    for word in missedWords:
        cpt += og_dict[word]
    
    score = ((total_mots - cpt)/total_mots)*100
    if record:
        saveData(audioPath, model_size + "/accuracy_score", score)
    else : 
        # print("Mots manquants dans la transcription de fw :", missedWords)
        print("Score of audio transcriptions of '%s' : %d/%d = %.2f" % (getFileName(audioPath), (total_mots - cpt), total_mots, score))

    return score


def processAudio(audioPath, fw_model, record=False):
    '''
    Takes an audio file, makes the transcription with faster-whisper, saves the
    transcription into a file and gives it a score that is saved into accuracy_score.txt file

    :params audioPath : str - file path of the audio file to process
    :params record : bool - if True it writes the data into a file, else just prints it on the terminal
    :return void
    '''
    getTranscript(audioPath, fw_model, record)
    getScore(audioPath, None, fw_model, record)
    print("Finished avualiting : '%s'" % getFileName(audioPath))

def processAudiowNoise(audioPath, fw_model, record=False):
    '''
    Takes an audio file with voice percentage, makes the transcription with faster-whisper, saves the
    transcription into a file and gives it a score that is saved into accuracy_score.txt file

    :params audioPath : str - file path of the audio file to process
    :params record : bool - if True it writes the data into a file, else just prints it on the terminal
    :return void
    '''
    getTranscript(audioPath, fw_model, record)
    name = re.split(r"[-/.]",audioPath)[-3]
    og_file = "transcriptions/original/og_" + name +".txt"
    getScore(audioPath, og_file, fw_model, record=True)
    print("Finished avualiting : '%s'" % getFileName(audioPath))


def processAllAudio(fw_model, directory = "samples"):
    '''
    Processes of all the audio files found in 'directory', the data would be automatically be 
    recorded into the files 'exec_time.txt' and 'accuracy_score.txt'

    :params directory : str - directory path where the audio files are located
    :return void
    '''
    data_dir = "data/" + fw_model
    if os.path.exists(data_dir + "/exec_time.txt") :
        os.remove(data_dir + "/exec_time.txt")
    if os.path.exists(data_dir + "/comparison_exec_time.txt") :
        os.remove(data_dir + "/comparison_exec_time.txt")
    if os.path.exists(data_dir + "/accuracy_score.txt"):
        os.remove(data_dir + "/accuracy_score.txt")

    files = glob.glob(directory + "/*.m4a") + glob.glob(directory + "/*.mp3")
    for audioPath in files :
        processAudio(audioPath, fw_model, record=True)
    
    files = glob.glob("samples/withNoise/*.mp3")
    for audioNoise in files:
        processAudiowNoise(audioNoise, fw_model, record=True)

def getFileName(filePath):
    '''
    Returns the name of the audio file (with noise level indicator)
    '''
    return re.split(r"[/.]",filePath)[-2]


# LLM processing stuff #

def testCodasLLM(audioPath, policy, model = "llama3.2:3b"):
    '''
    Takes a audio file and inputs the trasncription done by faster-whisper to the chat to ollama with the 
    predeterminated model 'llama3.2:3b'

    :params content : - str : content given to the llm 
    :parans model : str - model name of the llm to be used to process the msg
    :returns answer : str - answer of the llm
    '''

    command = getTranscript(audioPath, record=False)
    file = open("coda.txt", "r")
    capabilities = file.read()

    messages = [
        {
            'role' : 'user',
            'content' : f'You are a robot controlled by the following Python class (your CODA policy):  {capabilities} '
                f'Command "{command}". '
                'What action(s) do you take? Respond with only the necessary function calls in Python-like syntax.',
                # "If you don't understant the user you can ask for clarifications, and specifying what needs to be clarified.",
        },
    ]

    response = chat(model, messages=messages)
    answer = response['message']['content']
    return answer


# Plotting stuff #

def plotScore():
    # TODO modify to take into account fw model
    score = {noise : [] for noise in range(0, 101, 10)}
    
    withNoise = glob.glob("samples/withNoise/*.mp3")
    noNoise = []
    for f in withNoise:
        noise = int(re.split(r"[-/.]",f)[-2])
        name = re.split(r"[-/.]",f)[-3]
        og_file = "transcriptions/original/og_" + name +".txt"
        currentScore = getScore(f, og_file, record=False)
        score[noise].append(round(currentScore,2))

        if name not in noNoise :
            noNoise.append(name)

    for name in noNoise:
        audioPath = "samples/"+ name +".m4a"
        og_file = "transcriptions/original/og_"+ name+".txt"
        currentScore = getScore(audioPath,og_file, record=False)
        score[0].append(round(currentScore, 2))

    print(score)

    data = []
    q1 = []
    q3 = []
    for i in range(0, 101, 10):
        data.append(np.mean(score[i]))
        q1.append(np.quantile(score[i],0.25))
        q3.append(np.quantile(score[i],0.75))


    # Plot
    plt.figure(figsize=(7,7))
    xvalues = range(0,101,10)
    plt.title("Score mean by noise percentage")
    plt.xlabel("Percentage of noise volume")
    plt.ylabel("Transcription score out of 100")
    plt.plot(xvalues, data, marker='o', linestyle='-')
    plt.fill_between(xvalues, q1, q3, alpha = 0.2)
    plt.grid(True, alpha=0.2)
    # plt.show()

def plotTimeComp():
    # TODO modify to take into account fw model
    data = np.loadtxt("data/comparison_exec_time.txt")
    transcribe_t = [t[0] for t in data]
    total_t = [t[1] for t in data]
    
    plt.figure(figsize=(7,7))
    plt.scatter(transcribe_t, total_t, alpha=0.4)
    plt.title("Rapport of execution time")
    plt.xlabel("Transcribe time (s)")
    plt.ylabel("Total time (s)")
    plt.grid(True, alpha=0.3)
    # plt.show()


# MAIN #

def main():
    # Processes only one audio file, predetermined to have record = False to not polluate the data files, will print data in terminal
    # fw_model_size = "tiny"
    fw_model_size = "small"
    # fw_model_size = "large-v3"
    # processAudio("samples/assignment.m4a", fw_model_size)
    # processAudio("samples/withNoise/calcul_coffee10.mp3")

    # Processes all the files in 'samples' directory
    processAllAudio(fw_model_size)

    # getTranscript("samples/juin.m4a", record=False)

    # Plotting data
    # plotScore()
    # plotTimeComp()
    # plt.show()

    # Test CODA's Policy
    # answer = testCodasLLM("samples/snack.m4a", "codas_policy.txt")
    # print("\nTest CODA with ollama\t =============")
    # print(answer)

    print("Finished.")
    


if __name__ == "__main__":
    main()