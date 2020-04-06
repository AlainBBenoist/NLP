import os
import time
import json
import pickle
import logging

import nltk

from loader import CorpusLoader
from reader import PickledCorpusReader

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from models import create_pipeline, score_models, fit_models, save_models, model_name, print_results
    
if __name__ == '__main__':
    results_file = "results.json"
    labels = ["artistic_event", "other_event", ]

    logging.basicConfig(level=logging.DEBUG)

    logging.debug('Start of Program')
    
    # Initializing corpus reader and loader (generates K-Folds)
    reader = PickledCorpusReader('./pickle_corpus')
    loader = CorpusLoader(reader, 5, shuffle=True, categories=labels)

    # Initalizing models 
    models = []
    for form in (LogisticRegression, SGDClassifier):
        models.append(create_pipeline(form(), True))
        models.append(create_pipeline(form(), False))

    models.append(create_pipeline(MultinomialNB(), False))
    models.append(create_pipeline(GaussianNB(), True))

    # Running all models
    os.remove(results_file)
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
    for model in models :
        key = input('Save '+model_name(model)+' (y/n):')
        if key in ['o', 'O', 'y', 'Y', ] :
            save_models([ model, ], 'model-')
