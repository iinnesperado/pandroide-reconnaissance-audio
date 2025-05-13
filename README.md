# Projet Androide
Projet réalisé dans le cadre du M1 Androide (AI2D)

**Sujet :** Reconnaissance audio par des modèles de fondation pour un robot manipulateur mobile.

## Fonctionnalités
Ce projet implémente un système de reconnaissance vocale en utilisant des modèles de LLM. Le système :
- Transcrit la parole en utilisant les modèles Faster-Whisper
- Traite les commandes via un LLM

## Structure
### Excécutables
- ```whisper_processor.py``` :  chargé de faire les transcription, on retrouve aussi les tests de performance des différents modèles de Faster-Whisper.
- ```llm_integration.py``` : on retrouve les tests d'interaction simples avec ollama ainsi que les tests _Code as Policy_ et _RAG_.
- ```main.py``` : responsable de faire les tests d'interaction synchrone et asynchrone.

## Utilisation
### Faire des transcription
Dans le main du fichier ```whisper_processor.py```, il est possible d'obtenir la transcription d'un des fichier audio disponibles dans le répertoire samples. Décommentez la ligne de code suivante en choisissant le modèle de Faster-Whisper :
````
fw_model_size = "large-v3"
getTranscript("samples/juin.m4a", fw_model_size, record=False)
````
Ceci permettra de faire la transcription du fichier audio ```juin.m4a``` avec le modèle "large-v3". Il est aussi possible de traiter l'audio et obtenir le score de la transcription à travers la fonction ```getScore()```.

### Pour lancer Ollama
Il est nécessaire de faire un pull sur le modèle ```3.2:3b``` et ```mxbai-embed-large``` en faisant sur le terminal:
```
ollama pull <model name>
```

### Faire les tests avec Ollama
Pour faire le test _Code as Policy_ ou _RAG_ il suffit de décommenter la fonction correspondante dans le main du fichier ```llm_integration.py```.
```
if __name__ == "__main__":
    # test_RAG()
    # test_CAD()
```
### Pour exécuter l'interaction avec Ollama
Lancez le fichier ```main.py``` pour tester l'interaction en temps réel avec Ollama. Le premier enregistrement réalisé en cliquant sur la touche b enregistre la requête. Puis tous les enregistrements qui suivraient permettraient de donner à Ollama des informations supplémentaires. 

**Disclaimer** : Il est nécessaire de donner la permission à VSCODE (ou votre IDE de préférence) l'accès au clavier pour réaliser ce test.

### Répertoires
```
./
├── src/
│   ├── main.py
│   ├── whisper_processor.py
│   └── llm_integration.py
├── data/
│   ├── tiny/
│   │   ├── exec_time.txt
│   │   └── comparison_exec_time.txt
│   ├── small/
│   │   ├── exec_time.txt
│   │   └── comparison_exec_time.txt
│   ├── medium/
│   │   ├── exec_time.txt
│   │   └── comparison_exec_time.txt
│   └── large-v3/
│       ├── exec_time.txt
│       └── comparison_exec_time.txt
├── transcriptions/
│   ├── tiny/
│   ├── small/
│   ├── medium/
│   └── large-v3/
├── samples/
│   ├── withNoise/
│   └── audio files (.m4a, .mp3)
├── code_as_policy.txt
└── README.md
```

Le répertoire ```samples``` contient les fichier audio utilisé pour les tests faits. Le répertoire ```samples/withNoise``` contient plus spécifiquement les fichiers audio avec les différents niveau de bruit (%) sous le format ```fileName-noiseLevel.mp3```.

Puis les données générées par les tests sont sauvegardées dans les répertoires ```transcriptions``` et ```data``` selon le modèle de Faster-Whisper utilisé.

## Dépendances
- faster-whisper
- ollama
- pyaudio
- numpy
- matplotlib
- pynput

## Contributeurs 
- Ines RAHAOUI
- Inés Tian RUIZ-BRAVO PLOVINS

## Encadrants
- Stéphane DONCIEUX
- Emiland GARRABE

## Liens utiles
- [Détails du projet](https://androide.lip6.fr/node/743)
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Ollama](https://github.com/ollama/ollama)
