
import collections
import glob
import re

fwFile = open("data/fw_scripts.txt", "r")
fw_read = fwFile.read()

ogFile = open("data/og_transcripts.txt", "r")
og_read = ogFile.read()

samplesName = glob.glob("samples/*.m4a")

fw_file = re.split(" |, |\t|\n", fw_read)
og_file= re.split(" |, |\t|\n", og_read)
#print(og_file[:200]

score = dict()

i = 2
for name in samplesName:
    try:
        fw_text = fw_file[fw_file.index(name)+1:fw_file.index('(%d)'%i)]
        og_text = og_file[og_file.index(name)+1:og_file.index('(%d)'%i)]
    except:
        fw_text = fw_file[fw_file.index(name)+1:]
        og_text = og_file[og_file.index(name)+1:]
    total = len(og_text)
    print("name file : ", name)
    print("mots dans fw :",len(fw_text))
    print("mots dans og :", total)
    '''for word in fw_text:
        if word in og_text:
            cpt += 1
    '''

    #cpt = total
    
    fw = collections.Counter(fw_text)
    og = collections.Counter(og_text)
    print("this", (set(og) - set(fw)))
    '''for word in fw:
        if word in og:
            cpt = cpt - max(fw[word] - og[word], 0)'''
    cpt = total - len(set(og_text) - set(fw_text))
    
    score[name] = cpt/total
    i += 1


#print("score : ", score)

scoreFile = open("data/accuracy.txt", "w")
scoreFile.write("# 1 - file name\n")
scoreFile.write("# 2 - same word/nb total mots %\n")
for sample in score:
    scoreFile.write("%s\t" % sample)
    scoreFile.write("%.2f\n" % (score[sample]*100))