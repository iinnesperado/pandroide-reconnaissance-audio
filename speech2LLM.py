from faster_whisper import WhisperModel
from ollama import chat
import time
import collections
import re
import os
import glob

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

    text = ""
    for segment in segments:
        # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        text += segment.text

    end = time.time()
    execTime = end - start
    if record:
        saveData(audioPath, "exec_time", execTime)
    else :
        print("Transcription of audio '%s' took : %.2fs" % (getFileName(audioPath),execTime))
    
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
    file.write(audioPath + "\t%.2f\n" % (data))
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
    ogFile = open(og_file, "r")
    og_read = ogFile.read()
    ogFile.close()

    # On transforme le texte en liste de mots
    fw_text = re.split(r"[,.\s\t\n]\s*", fw_read)
    og_text = re.split(r"[,.\s\t\n]\s*", og_read)

    total_mots = len(og_text)

    # missedWords donne les mots qui ne sont pas dans fw_text par rapport à og_text
    missedWords = set(og_text) - set(fw_text)
    # print("Mots manquants dans la transcription de fw :", missedWords)

    # Transforme la liste de mots og_text en dictionaire de fréquences de mots
    # pour savoir le nombre total de mots qui manquent (cas où un mots serait plusieurs fois
    # mal compris)
    og_dict = collections.Counter(og_text)
    cpt = 0 
    for word in missedWords:
        cpt += og_dict[word]
    
    score = ((total_mots - cpt)/total_mots)*100
    if record:
        saveData(audioPath, "accuracy_score", score)
    else : 
        print("Score of audio transcriptions of '%s' : %.2f" % (getFileName(audioPath), score))

    return score


def processAudio(audioPath, og_file = '', record=False):
    '''
    Takes an audio file, makes the transcription with faster-whisper, saves the
    transcription into a file and gives it a score that is saved into accuracy_score.txt file

    :params audioPath : str - file path of the audio file to process
    :params of_file : str - used for cases when the og file has a different name to the fw file as it exist different audio versions of the same text (ambient noise differences)
    :params record : bool - if True it writes the data into a file, else just prints it on the terminal
    :return void
    '''

    getText(audioPath, record)
    
    if og_file == '' :
        og_file = "transcriptions/og_"+getFileName(audioPath)+".txt"
    getScore(audioPath,og_file, record)
    print("Finished avualiting audio file : '%s'" % getFileName(audioPath))

def processAllAudio(directory = "samples"):
    '''
    Processes of all the audio files found in 'directory', the data would be automatically be 
    recorded into the files 'exec_time.txt' and 'accuracy_score.txt'

    :params directory : str - directory path where the audio files are located
    :return void
    '''
    if os.path.exists("data/exec_time.txt") :
        os.remove("data/exec_time.txt")
    if os.path.exists("data/accuracy_score.txt"):
        os.remove("data/accuracy_score.txt")

    files = glob.glob(directory + "/*.m4a") + glob.glob(directory + "/*.mp3")
    pass
    for audioPath in files :
        processAudio(audioPath,og_file='', record=True)

def getFileName(filePath):
    return re.split(r"[/.]",filePath)[-2]


# LLM processing stuff #

def getResponse(audioPath, model = "llama3.2:3b"):
    '''
    Takes a audio file and inputs the trasncription done by faster-whisper to the chat to ollama with the 
    predeterminated model 'llama3.2:3b'

    :params content : - str : content given to the llm 
    :parans model : str - model name of the llm to be used to process the msg
    :returns answer : str - answer of the llm
    '''

    content = getText(audioPath,record=False)

    messages = [
        {
            'role' : 'user',
            'content' : content,
        },
    ]

    response = chat(model, messages=messages)
    answer = response['message']['content']
    return answer

# Plotting stuff #
# NOTE end time to record exec time was put after text was saved into a var bc this step also took some time and so we believed it 
# was more interesting to record that since its something to take into account with ollama implementation 
# TODO plot exec time compared to 

# MAIN #

def main():
    # Processes only one audio file, recommended to have record = False to not polluate the data files, it will then print
    # the result in the terminal. Parameter og_file is only used when a file has different versions of ambient noise since
    # faster-whisper would give different files but the original transcription is the same.
    # processAudio("samples/calcul.m4a",'', record=False)
    # processAudio("samples/withNoise/calcul_coffee10.mp3","transcriptions/og_calcul.txt", record=False)


    # Processes all the files in 'samples' directory
    # processAllAudio("samples")

    answer = getResponse("samples/withNoise/calcul_coffee10.mp3",model="llama3.2:3b")
    print(answer)

    print("Finished.")
    


if __name__ == "__main__":
    main()