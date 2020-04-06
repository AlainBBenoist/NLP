import time
import json

import unicodedata
import pickle
import tabulate

import numpy as np
import nltk

from loader import CorpusLoader
from reader import PickledCorpusReader

from nltk import sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# This function returns its arguments without transformation
def identity(words):
    return words

# TextNormalizer - a custom text normalization transformer
# See page 72 of Applied Text Analysis with Python
class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='french'):
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        return [
            self.lemmatize(token, tag).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token)
        ]

    def lemmatize(self, token, pos_tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0] if len(pos_tag) > 0 else 'X', wn.NOUN)
        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.normalize(document[0])
            
# See page 90 of Applied Text Analysis with Python
# This will create a pipeline with a given model provided as the 'estimator' parameter
def create_pipeline(estimator, reduction=False):

    steps = [
        ('normalize', TextNormalizer()),
        ('vectorize', TfidfVectorizer(
            tokenizer=identity, preprocessor=None, lowercase=False
        ))
    ]

    if reduction:
        steps.append((
            'reduction', TruncatedSVD(n_components=1000)
        ))

    # Add the estimator
    steps.append(('classifier', estimator))
    return Pipeline(steps)

def model_name(model) :
    """
    Returns a model name
    """
    name = model.named_steps['classifier'].__class__.__name__
    if 'reduction' in model.named_steps:
        name += " (TruncatedSVD)"
    return name

# Running and evaluate Models
# Page 93 of Applied Text Analysis with Python
def score_models(models, loader):
    for model in models:

        name = model_name(model)
        scores = {
            'model': str(model),
            'name': name,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'time': [],
        }

        for X_train, X_test, y_train, y_test in loader:
            ndocs = ncats = 0
            start = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            scores['time'].append(time.time() - start)
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))

        yield scores

# Details by labels for a given estimator
def print_classification(estimator, reduction=False) :
    from sklearn.metrics import classification_report
    
    loader = CorpusLoader(reader, 2, shuffle=True, categories=labels)
    model = create_pipeline(estimator, reduction)
    for X_train, X_test, y_train, y_test in loader:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, labels=labels))

def fit_models(models, reader, categories=None) :
    """
    Fitting a model on the entire dataset
    """
    for model in models :
        print(model_name(model))
        X_train = [ list(reader.docs(fileids=[fileid]))
                        for fileid in reader.fileids(categories=categories) ]
        y_train = [ reader.categories(fileids=[fileid])[0]
                        for fileid in reader.fileids(categories=categories) ]
        model.fit(X_train, y_train)
    return True

def save_models(models, prefix) :
    for model in models :
        name = model_name(model)
        key = input(name+' (o/n): ')
        if key in ['o', 'O', ] :
            with open(prefix+name+'.pickle', 'wb') as f:
                pickle.dump(model, f)

def predict_model(model, X_test) :
    y_pred = model.predict(X_test)
    print(model_name(model))
    print(y_pred)
    return y_pred

def print_results(results_file) :
    """
    Print results of model scores
    """
    fields = ['model', 'precision', 'recall', 'accuracy', 'f1']
    table = []

    with open('results.json', 'r') as f:
        for idx, line in enumerate(f):
            scores = json.loads(line)

            row = [scores['name']]
            for field in fields[1:]:
                row.append("{:0.3f}".format(np.mean(scores[field])))

            table.append(row)

    table.sort(key=lambda r: r[-1], reverse=True)
    print(tabulate.tabulate(table, headers=fields))
    
if __name__ == '__main__':
    results_file = "results.json"
    labels = ["artistic_event", "other_event", ]

    # Initialzing corpus reader and loader (generates K-Folds)
    reader = PickledCorpusReader('./pickle_corpus')
    loader = CorpusLoader(reader, 5, shuffle=True, categories=labels)

    txt=TextNormalizer()
    txt.lemmatize("qu'","")

    # Initalizing models 
    models = []
    for form in (LogisticRegression, SGDClassifier):
        models.append(create_pipeline(form(), True))
        models.append(create_pipeline(form(), False))

    models.append(create_pipeline(MultinomialNB(), False))
    models.append(create_pipeline(GaussianNB(), True))

    # Running all models
    for scores in score_models(models, loader):
        with open(results_file, 'a') as f:
            f.write(json.dumps(scores) + "\n")
    # And print results
    print_results(results_file)
    
    # Added later
    #print_classification(SGDClassifier(), False)

    # Fit models with all data
    fit_models(models, reader, categories=labels)
    
    # Save models
    save_models(models, 'model-')
