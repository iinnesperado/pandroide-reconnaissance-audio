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

# Audio processing stuff #

def getText(audioPath, record):
    '''
    Fait la transcription du fichier audio donnée
    
    :param audioPath : str - path to audio file (m4a or mp3)
    :params record : bool - if True it writes the data into a file, else just prints it
    :return text : str - transcription of the audio
    :return execTime : float
    '''

    model_size = "large-v3"

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

    text = ""
    for segment in segments:
        # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        text += segment.text

    end2 = time.time()
    total_t = end2 - start
    if record:
        saveData(audioPath, "exec_time", total_t)
        # to plot a comparison of exec time of func transcribe and the time with saving text
        file = open("data/comparison_exec_time.txt", "a")
        file.write("%.2f\t%.2f\n" % (transcribe_t, total_t))
        file.close()
    else :
        print("Transcription of audio '%s' took : %.2fs" % (getFileName(audioPath),total_t))
    
    saveText(text,"transcriptions/fw_"+ getFileName(audioPath) + ".txt")

    return text

def saveText(text, filePath):
    '''
    Creates a file and saves the transcriptions of the audioPath done by faster-whisper
    as well as record the execution time into ecxec_time.txt file
    
    :param text : str - content (text) to save into a file
    :param filePath : str - file path to the file where the transcriptions will be saved
    :return void
    '''

    file = open(filePath, "w")
    file.write(text)
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

def getScore(audioPath, og_file, record):
    '''
    Gives accuracy score to the transcription done by faster-wshiper out of 100.
    Score = num of words missing in fw_text (compared to og_text)/ total words in og_text
    
    :params audioPath : str - path to audio file (m4a or mp3)
    :param  og_file : str - file path to transcription
    :params record : bool - if True it writes the data into a file, else just prints it
    :return score : float
    '''
    # Transcription faite par faster whisper
    fw_file = "transcriptions/fw_"+getFileName(audioPath)+".txt"
    fwFile = open(fw_file, "r")
    fw_read = fwFile.read()
    fwFile.close()

    # Transcription faites par nous utilisée comme référence
    if og_file==None:
        og_file = "transcriptions/og_"+getFileName(audioPath)+".txt"
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
        saveData(audioPath,"accuracy_score", score)
    else : 
        # print("Mots manquants dans la transcription de fw :", missedWords)
        print("Score of audio transcriptions of '%s' : %d/%d = %.2f" % (getFileName(audioPath), (total_mots - cpt), total_mots, score))

    return score


def processAudio(audioPath, record=False):
    '''
    Takes an audio file, makes the transcription with faster-whisper, saves the
    transcription into a file and gives it a score that is saved into accuracy_score.txt file

    :params audioPath : str - file path of the audio file to process
    :params record : bool - if True it writes the data into a file, else just prints it on the terminal
    :return void
    '''
    getText(audioPath, record)
    getScore(audioPath, None, record)
    print("Finished avualiting : '%s'" % getFileName(audioPath))

def processAudiowNoise(audioPath, record=False):
    '''
    Takes an audio file with voice percentage, makes the transcription with faster-whisper, saves the
    transcription into a file and gives it a score that is saved into accuracy_score.txt file

    :params audioPath : str - file path of the audio file to process
    :params record : bool - if True it writes the data into a file, else just prints it on the terminal
    :return void
    '''
    getText(audioPath, record)
    name = re.split(r"[-/.]",audioPath)[-3]
    og_file = "transcriptions/og_" + name +".txt"
    getScore(audioPath, og_file, record=True)
    print("Finished avualiting : '%s'" % getFileName(audioPath))


def processAllAudio(directory = "samples"):
    '''
    Processes of all the audio files found in 'directory', the data would be automatically be 
    recorded into the files 'exec_time.txt' and 'accuracy_score.txt'

    :params directory : str - directory path where the audio files are located
    :return void
    '''
    if os.path.exists("data/exec_time.txt") :
        os.remove("data/exec_time.txt")
    if os.path.exists("data/comparison_exec_time.txt") :
        os.remove("data/comparison_exec_time.txt")
    if os.path.exists("data/accuracy_score.txt"):
        os.remove("data/accuracy_score.txt")

    files = glob.glob(directory + "/*.m4a") + glob.glob(directory + "/*.mp3")
    for audioPath in files :
        processAudio(audioPath, record=True)
    
    files = glob.glob("samples/withNoise/*.mp3")
    for audioNoise in files:
        processAudiowNoise(audioNoise, record=True)

def getFileName(filePath):
    '''
    Returns the name of the audio file (without noise)
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

    command = getText(audioPath, record=False)
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
    score = {noise : [] for noise in range(0, 101, 10)}
    
    withNoise = glob.glob("samples/withNoise/*.mp3")
    noNoise = []
    for f in withNoise:
        noise = int(re.split(r"[-/.]",f)[-2])
        name = re.split(r"[-/.]",f)[-3]
        og_file = "transcriptions/og_" + name +".txt"
        currentScore = getScore(f, og_file, record=False)
        score[noise].append(round(currentScore,2))

        if name not in noNoise :
            noNoise.append(name)

    for name in noNoise:
        audioPath = "samples/"+ name +".m4a"
        og_file = "transcriptions/og_"+ name+".txt"
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
    # Processes only one audio file, predetermined to have record = False to not polluate the data files, 
    # it will then print the result in the terminal
    # processAudio("samples/assignment.m4a")
    # processAudio("samples/withNoise/calcul_coffee10.mp3")

    # Processes all the files in 'samples' directory
    # processAllAudio("samples")

    # getText("samples/juin.m4a", record=False)

    # Plotting data
    # plotScore()
    # plotTimeComp()
    # plt.show()

    # Test CODA's Policy
    answer = testCodasLLM("samples/snack.m4a", "codas_policy.txt")
    print("\nTest CODA with ollama\t =============")
    print(answer)

    print("Finished.")
    


if __name__ == "__main__":
    main()