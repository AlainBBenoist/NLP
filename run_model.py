import codecs
import pickle
import csv
import logging

from models import create_pipeline, score_models, fit_models, predict_model, save_models, model_name, print_results, TextNormalizer, identity

def print_event(doc) :
    for para in doc : 
        for sent in para :
            print(' '.join(token for token, _ in sent))
    
def main() :
    """
    processing text
    """
    from reader import PickledCorpusReader
    
    # Classification model
    model_file='model-SGDClassifier.pickle'

    # Initialize model from pickle 
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Initialize a corpus reader
    corpus = PickledCorpusReader('./pickle_corpus/')

    # Call the model to classify text
    y_pred, x_pred = predict_model(model, corpus, categories=['artistic_event'])

    # Print result
    count = nartistics = 0
    for result in y_pred :
        if result != 'artistic_event' :
            for doc in x_pred[count] :
                print_event(doc)                
            print('======================')
            nartistics += 1
        count += 1
    print('{:d} artistic events found/ {:d} events'.format(nartistics, count))

def read_events(filename) :
    """
    Read all events contained in a csv file and loads them in memory
    """
    events = list()
    event_id = 1
    with codecs.open(filename, 'r', "utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        headers=reader.fieldnames
        for row in reader :
            if len(row['description']) == 0 :
                continue
            event = {
                'id' : event_id,
                'name' : row['name'],
                'description' : row['description'],
                'eventcat' : row['class'],
            }
            events.append(event)
            event_id += 1
    csv_file.close()
    return events

def main2() :
    from event_reader import EventCorpusReader

    """
    processing text
    """
    
    event1 = {  'id'    : 1,
                'name' : 'Exposition de peinture',
                'description' : 'ceci est une exposition de Claude Monet.\nVernissage le 23 mai 2020',
                'eventcat' : [ 'artistic_event' ]
            }
    event2 = {  'id'    : 2,
                'name' : 'Exposition de sculpture',
                'description' : 'Ceci est une exposition de Rodin',
                'eventcat' : ['artistic_event'],
            }
    event3 = {  'id'    : 3,
                'name' : 'Fête du village à Roquamadour',
                'description' : 'Animation, buvette, musique et attractions diverses',
                'eventcat' : ['artistic_event'],
            }
    events = [ event1, event2, event3 ]

    # Read the events file
    events = read_events('events.csv')
##    count = 0
##    for event in events :
##        print(event)
##        count += 1
##        if count > 10 :
##            break
    print('{:d} events loaded'.format(len(events)))

    # Initialize a corpus reader    
    corpus = EventCorpusReader(events, categories=['artistic-event'])
##    for fileid in corpus.fileids() :
##        print(fileid)

    # Classification model
    model_file='model-SGDClassifier.pickle'

    # Initialize model from pickle 
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Call the model to classify text
    y_pred, x_pred = predict_model(model, corpus, categories=['artistic-event'])

    # Print result
    count = nartistics = 0
    for result in y_pred :
        if result != 'artistic_event' :
            for doc in x_pred[count] :
                print_event(doc)                
            print('======================')
            nartistics += 1
        count += 1
    print('{:d} artistic events found/ {:d} events'.format(nartistics, count))
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
    main2()
