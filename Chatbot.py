# Libraries needed for NLP
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Libraries needed for Tensorflow processing
import tensorflow as tf
import numpy as np
import tflearn
import random
import json

from google.colab import files
files.upload()

# import our chat-bot intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)

intents

words = []
classes = []
documents = []
ignore = ['?']
