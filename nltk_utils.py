from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np

nltk.download('punkt')

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tok_sentence, all_words):
    tok_sentence = [stem(w) for w in tok_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, w in enumerate(all_words):
        if w in tok_sentence:
            bag[i] = 1.0
    return bag
