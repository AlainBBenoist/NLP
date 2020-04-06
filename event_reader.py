#!/usr/bin/env python3

import codecs
import unicodedata
import time
import pickle
import logging
import bs4

import nltk
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

from readability.readability import Unparseable
from readability.readability import Document as Paper
from nltk import sent_tokenize, wordpunct_tokenize #, pos_tag

# Spacy Model (spacy)
import spacy
from spacy import displacy

from pos_tagger import pos_tagger

# TODO :
# Create a pos_tag module which encapsulates the various pos taggers
# Handle multiple paragraphs in descriptio of events 

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']

class EventCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    A corpus reader for events to enable preprocessing.
    """

    def __init__(self, events, fileids=None, encoding='utf8',
                 tags=TAGS, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining
        arguments are passed to the ``CorpusReader`` constructor.
        """
        # Add the default category pattern if not passed into the class.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = None

        # Initialize the NLTK corpus reader objects
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, '.', fileids, encoding)

        # Save the events list
        self.events = events
        self.tagger = pos_tagger('spacy')
        self.htmltags = tags

    def resolve(self, fileids, categories):
        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. Implemented similarly to
        the NLTK ``CategorizedPlaintextCorpusReader``.
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories=categories)
        if fileids is not None :
            return fileids
        fileids = []

        # Nothing specified means we return all event identifiers
        for event in self.events :
            fileids.append(event['id'])
        return fileids
    
    def fileids(self, categories=None) :
        """
        Returns a list of event_ids corresponding to categories
        """
        # Construct a list of fileids
        fileids = []
        for event in self.events :
            if categories :
                # check that event belongs to the given list of categories 
                if 'eventcat' in event and event['eventcat'] :
                    # Determine the intersection between categories of event and required categories
                    intersect = [value for value in categories if value in event['eventcat']]
                    if len(intersect) > 0 :
                        fileids.append(event['id'])
            else :
                # Everything will be retrieved
                fileids.append(event['id'])
        return fileids

    def get_event(self, event_id) :
        """
        Access to an event by event_id
        """
        for event in self.events :
            if event['id'] == event_id :
                return event
        return None

    def docs(self, fileids=None, categories=None):
        """
        Returns the events after pos tagging them
        """
        # Initialize pos tagger if necessary - using spacy
        #if self.tagger is None :
        #    self.tagger = spacy.load('fr_core_news_sm')

        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)
        #print(fileids)

        # Retrieve events
        for fileid in fileids :
            paras = []
            event = self.get_event(fileid)
            assert(event is not None)
            # Build paragraphs
            # Limitation : one paragraph for the description
            paras.append(event['name'])
            html = Paper(event['description']).summary()
##            print(event['description'])
##            print(html)
            soup = bs4.BeautifulSoup(html, 'lxml')
            for element in soup.find_all(self.htmltags):
                #print('§'+element.text)
                paras.append(element.text)
            soup.decompose()
##            print(paras)
##            key=input('>>')
##            for item in ['name', 'description', ] :
##                paras.append(event[item])
                
            # Return tags
            yield [ [ self.tagger.pos_tag(sent) for sent in sent_tokenize(para) ]
                    for para in paras ]
            
##            yield [ [ [ (token.text, token.pos_) for token in self.tagger(sent) ]
##                        for sent in sent_tokenize(para) ]
##                    for para in paras ]           

    def sizes(self, fileids=None, categories=None):
        """
        Returns a list of tuples, the fileid and size on disk of the file.
        This function is used to detect oddly large files in the corpus.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, getting every path and computing filesize
        for event in self.events :
            yield len(event['name']) + len(event['description'])

    def paras(self, fileids=None, categories=None):
        """
        Get paragraphs from an event 
        """
        for doc in self.docs(fileids, categories) :
            for paragraph in doc :
                yield paragraph


    def sents(self, fileids=None, categories=None):
        """
        Uses the built in sentence tokenizer to extract sentences from the
        paragraphs. Note that this method uses BeautifulSoup to parse HTML.
        """
        for paragraph in self.paras(fileids, categories):
            for sentence in paragraph :
                #print('$'+sentence)
                yield sentence

    def words(self, fileids=None, categories=None):
        """
        Uses the built in word tokenizer to extract tokens from sentences.
        Note that this method uses BeautifulSoup to parse HTML content.
        """
        for sentence in self.sents(fileids, categories):
            for token in sentence:
                yield token[0]

    def describe(self, fileids=None, categories=None):
        """
        Performs a single pass of the corpus and
        returns a dictionary with a variety of metrics
        concerning the state of the corpus.
        """
        started = time.time()

        # Structures to perform counting.
        counts  = nltk.FreqDist()
        tokens  = nltk.FreqDist()

        # Perform single pass over paragraphs, tokenize and count
        for para in self.paras(fileids, categories):
            counts['paras'] += 1

            for sent in sent_tokenize(para):
                counts['sents'] += 1

                for word in wordpunct_tokenize(sent):
                    counts['words'] += 1
                    tokens[word] += 1

        # Compute the number of files and categories in the corpus
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_topics  = len(self.categories(self.resolve(fileids, categories)))

        # Return data structure with information
        return {
            'files':  n_fileids,
            'topics': n_topics,
            'paras':  counts['paras'],
            'sents':  counts['sents'],
            'words':  counts['words'],
            'vocab':  len(tokens),
            'lexdiv': float(counts['words']) / float(len(tokens)),
            'ppdoc':  float(counts['paras']) / float(n_fileids),
            'sppar':  float(counts['sents']) / float(counts['paras']),
            'secs':   time.time() - started,
        }

if __name__ == '__main__':
    from collections import Counter
    from reader import PickledCorpusReader

    event1 = {  'id'    : 1,
                'name' : 'Exposition de peinture',
                'description' : 'ceci est une exposition de Claude Monet.\nVernissage le 23 mai 2020',
                'eventcat' : [ 200 ]
            }
    event2 = {  'id'    : 2,
                'name' : 'Exposition de sculpture',
                'description' : 'Ceci est une exposition de Rodin',
                'eventcat' : [200],
            }
    events = [ event1, event2 ]
    fileid='artistic_event/0b7944a1a37ff80d982f0e18abeab26d.pickle'
    corpus1 = PickledCorpusReader('./pickle_corpus', fileids=[fileid])#categories='artistic_event') 
    corpus2 = EventCorpusReader(events, categories=[300])

    for corpus in [corpus1, corpus2] :
        print('------- words --------')
        count = 0 
        for word in corpus.words(categories=[200, 300]) :
            print(word)
            count += 1
            if count > 10 :
                break
        print('------- sents --------')
        count = 0
        for sent in corpus.sents(categories=[200, 300]) :
            print(sent)
            count += 1
            if count > 5 :
                break
        print('------- paras --------')
        count = 0
        for para in corpus.paras(categories=[200, 300]) :
            print(para)
            count += 1
            if count > 5 :
                break
        print('------- docs --------')
        count = 0
        for doc in corpus.docs(categories=[200, 300]) :
            print(doc)
            count += 1
            if count > 5 :
                break
                count = 0
        print('------- fileids --------')
        for fileid in corpus.fileids(categories=[200, 300]) :
            print(fileid)
            count += 1
            if count > 5 :
                break
        print('=========================')

