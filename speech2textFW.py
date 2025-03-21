from faster_whisper import WhisperModel
import time
import collections
import re
import os

'''
All mentions of 'data' refer to : 
    - execution time
    - accuracy score
'''

def speech2text(audioPath, recordData = True):
    '''
    Fait la transcription du fichier audio donnée
    
    :param audioPath : str - path to audio file (m4a or mp3)
    :params recordData : bool - if True it writes the data into a file, else just prints it
    :return script : str
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
    end = time.time()
    execTime = end - start
    if recordData:
        saveTime(audioPath, execTime)
    else :
        print("Transcription of audio '%s' took : %.2fs" % (getFileName(audioPath),execTime))

    script = ""
    for segment in segments:
        # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        script += segment.text

    saveScript(script,"scripts/fw_"+ getFileName(audioPath) + ".txt")


    return script

def saveScript(script, filePath):
    '''
    Creates a file and saves the script of the audioPath done by faster-whisper
    as well as record the execution time into ecxec_time.txt file
    
    :param script : str - content (text) to save into a file
    :param filePath : str - file path to the file where the script will be saved
    :return void
    '''

    file = open(filePath, "w")
    file.write(script)
    file.close()

def saveTime(audioPath, execTime):
    try:
        file = open("data/exec_time.txt", "a")
    except:
        file = open("data/exec_time.txt", "w")
    file.write(audioPath + "\t%.2f\n" % (execTime))
    file.close()

def giveScore(fw_file, og_file):
    '''
    Calcule le accuracy score de la transcription de faster-whisper sur 100.
    Score = nb de mots manquant en fw_text (par rapport à og_text)/ total mots de og_text
    
    :param  fw_file, og_file : str - file path du script 
    :return score : float
    '''
    
    # Transcription faite par faster whisper
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
    # print("Mots manquants dans la transcription de fw :", missedWords))

    # Transforme la liste de mots og_text en dictionaire de fréquences de mots
    # pour savoir le nombre total de mots qui manquent (cas où un mots serait plusieurs fois
    # mal compris)
    og_dict = collections.Counter(og_text)
    cpt = 0 
    for word in missedWords:
        cpt += og_dict[word]
    
    score = ((total_mots - cpt)/total_mots)*100
    # print(score)

    return score

def processAudio(audioPath, recordData = True):
    '''
    Takes an audio file, makes the transcript with faster-whisper, saves the
    transcript into a file and gives it a score that is saved into accuracy_score.txt file

    :params audioPath : str - file path of the audio file to process
    :params recordData : bool - if True it writes the data into a file, else just prints it
    :return void
    '''

    speech2text(audioPath, recordData)
    fw_file = "scripts/fw_"+getFileName(audioPath)+".txt"
    og_file = "scripts/og_"+getFileName(audioPath)+".txt"
    score = giveScore(fw_file,og_file)

    if recordData:
        # Saves scores in file data/accuracy_score.txt
        try:
            file = open("data/accuracy_score.txt","a")
        except:
            file = open("data/accuracy_score.txt","w")
        file.write(audioPath + "\t%.2f\n" % (score))
        file.close()
    else : 
        print("Score of audio script of '%s' : %.2f" % (getFileName(audioPath), score))
    print("Finished avualiting audio file : '%s'" % getFileName(audioPath))


def processAllAudio(directory = 'samples'):
    '''
    Processes of all the audio files found in 'directory', the data would be automatically be 
    recorded into the files 'exec_time.txt' and 'accuracy_score.txt'

    :params directory : str - directory path where the audio files are located
    :return void
    '''
    if os.path.exists("data/exec_time.txt") :
        os.remove("data/exec_time.txt")
    if os.path.exists("data_accuracy_score.txt"):
        os.remove("data/accuracy_score.txt")

    files = os.listdir(directory)
    for audioPath in files :
        if audioPath.endswith(('.m4a','.mp3')):
            processAudio(audioPath, recordData=True)

def getFileName(filePath):
    return re.split(r"[/.]",filePath)[-2]

def main():
    # Processes only one data, recommended to have recordData = False to not polluate the data files
    processAudio("samples/assignment.m4a", recordData=False)

    # Processes all the files in 'samples' directory
    # processAllAudio(directory='samples')
    print('Finished.')
    


if __name__ == "__main__":
    main()