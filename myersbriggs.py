import random
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_favorite_world(comment):
    L1 = ['I', 'E']
    vocab = pickle.load(open('MBTI_Vocab.pkl', 'rb'))
    loaded_vectorizer = CountVectorizer(vocabulary=vocab)
    loaded_vectorizer._validate_vocabulary()
    logregfw = pickle.load(open('favoriteworld.pkl', 'rb'))
    X = comment
    array = loaded_vectorizer.fit_transform([X]).toarray()
    favoriteworld = logregfw.predict(array)
    return "I" if favoriteworld.tolist().pop()==0 else "E"
    
def get_information(comment):
    L2  = ['N', 'S']
    vocab = pickle.load(open('MBTI_Vocab.pkl', 'rb'))
    loaded_vectorizer = CountVectorizer(vocabulary=vocab)
    loaded_vectorizer._validate_vocabulary()
    logreginfo = pickle.load(open('information.pkl', 'rb'))
    X = comment
    array = loaded_vectorizer.fit_transform([X]).toarray()
    information = logreginfo.predict(array)
    return "N" if information.tolist().pop()==0 else "S"

def get_decision(comment):
    L3  = ['T', 'F']
    vocab = pickle.load(open('MBTI_Vocab.pkl', 'rb'))
    loaded_vectorizer = CountVectorizer(vocabulary=vocab)
    loaded_vectorizer._validate_vocabulary()
    logregdecision = pickle.load(open('decision.pkl', 'rb'))
    X = comment
    array = loaded_vectorizer.fit_transform([X]).toarray()
    decision = logregdecision.predict(array)
    return "T" if decision.tolist().pop()==0 else "F"

def get_structure(comment):
    L4  = ['J', 'P']
    vocab = pickle.load(open('MBTI_Vocab.pkl', 'rb'))
    loaded_vectorizer = CountVectorizer(vocabulary=vocab)
    loaded_vectorizer._validate_vocabulary()
    logregstructure = pickle.load(open('structure.pkl', 'rb'))
    X = comment
    array = loaded_vectorizer.fit_transform([X]).toarray()
    structure = logregstructure.predict(array)
    return "J" if structure.tolist().pop()==0 else "P"

def mbti_predict(comment):
    indicator_1 = get_favorite_world(comment)
    indicator_2 = get_information(comment)
    indicator_3 = get_decision(comment)
    indicator_4 = get_structure(comment)

    macro_indicator = "{0}{1}{2}{3}".format(indicator_1, indicator_2, indicator_3, indicator_4)
    return macro_indicator