
import collections
import glob
import re

# Transcription faite par faster whisper de tous les fichiers audio
fwFile = open("data/fw_scripts.txt", "r")
fw_read = fwFile.read()

# Transcription faites par nous utilisée comme référence de tous les fichier audio
# Il est important que l'ordre des transcriptions faite à main soient
# dans le même ordre que les transcriptions faites par faster-whisper
ogFile = open("data/og_scripts.txt", "r")
og_read = ogFile.read()

directory = glob.glob('samples/*.m4a') + glob.glob('samples/*.mp3')

# On transforme les textes en liste de mots
fw_file = re.split(" |, |\t|\n", fw_read)
og_file= re.split(" |, |\t|\n", og_read)

# On store le score dans in dict('nom de fichier' : valeur du score dans [0,1])
score = dict()

i = 2
for name in directory:
    try:
        # On selectionne que la partie texte d'un seul fichier audio pour faire la 
        # comparaison et donner un score
        fw_text = fw_file[fw_file.index(name)+1:fw_file.index('(%d)'%i)]
        og_text = og_file[og_file.index(name)+1:og_file.index('(%d)'%i)]
    except:
        fw_text = fw_file[fw_file.index(name)+1:]
        og_text = og_file[og_file.index(name)+1:]
    total_mots = len(og_text)
    # print("Nom du fichier : ", name)
    # print("Nobre de mots dans fw :",len(fw_text))
    # print("Nombre de mots dans og :", total_mots)
    
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
    
    score[name] = (total_mots - cpt)/total_mots
    i += 1


#print("score : ", score)

scoreFile = open("data/accuracy_score.txt", "w")
scoreFile.write("# 1 - file name\n")
scoreFile.write("# 2 - same word/nb total mots %\n")
for sample in score:
    scoreFile.write("%s\t" % sample)
    scoreFile.write("%.2f\n" % (score[sample]*100))