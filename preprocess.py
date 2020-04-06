import os
import nltk
import pickle

from nltk import sent_tokenize, wordpunct_tokenize # , pos_tag

# Stanford POS Tagger 
from stanford_pos_tag import pos_tag

# Stanford Core NLP (stanford) 
from stanfordnlp.server import CoreNLPClient

# Spacy Model (spacy)
import spacy
from spacy import displacy
from pos_tagger import pos_tagger

model = 'default'     # Seems to be the most effective POS Tagger for french language - and the fastest

class Preprocessor(object):
    """
    The preprocessor wraps a corpus object (usually a `HTMLCorpusReader`)
    and manages the stateful tokenization and part of speech tagging into a
    directory that is stored in a format that can be read by the
    `HTMLPickledCorpusReader`. This format is more compact and necessarily
    removes a variety of fields from the document that are stored in the JSON
    representation dumped from the Mongo database. This format however is more
    easily accessed for common parsing activity.
    """

    def __init__(self, corpus, target=None, **kwargs):
        """
        The corpus is the `HTMLCorpusReader` to preprocess and pickle.
        The target is the directory on disk to output the pickled corpus to.
        """
        self.corpus = corpus
        self.target = target
        self.tagger = pos_tagger('spacy')
        
        # Modification for dibutade
        if model == 'stanford' :
            os.environ['CORENLP_HOME'] = 'C:/Users/alain/OneDrive/Ateliers Dibutade/NLP/stanford-corenlp-full-2018-10-05'
            self.pos_tagger = CoreNLPClient(properties='french', annotators=['pos', ], timeout=30000, memory='1G')
        elif model == 'spacy' :
            self.nlp = spacy.load('fr_core_news_sm')


    def fileids(self, fileids=None, categories=None):
        """
        Helper function access the fileids of the corpus
        """
        fileids = self.corpus.resolve(fileids, categories)
        if fileids:
            return fileids
        return self.corpus.fileids()

    def abspath(self, fileid):
        """
        Returns the absolute path to the target fileid from the corpus fileid.
        """
        # Find the directory, relative from the corpus root.
        parent = os.path.relpath(
            os.path.dirname(self.corpus.abspath(fileid)), self.corpus.root
        )

        # Compute the name parts to reconstruct
        basename  = os.path.basename(fileid)
        name, ext = os.path.splitext(basename)

        # Create the pickle file extension
        basename  = name + '.pickle'

        # Return the path to the file relative to the target.
        return os.path.normpath(os.path.join(self.target, parent, basename))

    def tokenize(self, fileid):
        """
        Segments, tokenizes, and tags a document in the corpus. Returns a
        generator of paragraphs, which are lists of sentences, which in turn
        are lists of part of speech tagged words.
        """
        for paragraph in self.corpus.paras(fileids=fileid):
            if model == 'original' :
                for sent in sent_tokenize(paragraph) :
                    print(sent)
                    print(wordpunct_tokenize(sent))
                    print(pos_tag(wordpunct_tokenize(sent)))
                key = input('Continue')
                yield [
                    pos_tag(wordpunct_tokenize(sent))
                    for sent in sent_tokenize(paragraph)
                ]
            elif model == 'stanford' :
                # Modification for the CORE NLP package
                ann = self.pos_tagger.annotate(paragraph)
                for sentence in ann.sentence :
                    #print(sentence)
                    #for token in sentence.token :
                    #    print((token.word, token.pos))
                    yield [[ (token.word, token.pos)
                            for token in sentence.token ]]
            elif model == 'spacy' :
                yield [
                    [ (token.text, token.pos_) for token in self.nlp(sent) ]
                    for sent in sent_tokenize(paragraph)
                ]
            else :  # Default - still to test
                for sent in sent_tokenize(paragraph) :
                    yield self.tagger.pos_tag(sent)
                    

    def process(self, fileid):
        """
        For a single file does the following preprocessing work:
            1. Checks the location on disk to make sure no errors occur.
            2. Gets all paragraphs for the given text.
            3. Segments the paragraphs with the sent_tokenizer
            4. Tokenizes the sentences with the wordpunct_tokenizer
            5. Tags the sentences using the default pos_tagger
            6. Writes the document as a pickle to the target location.
        This method is called multiple times from the transform runner.
        """
        
        # Compute the outpath to write the file to.
        target = self.abspath(fileid)
        parent = os.path.dirname(target)

        # Make sure the directory exists
        if not os.path.exists(parent):
            os.makedirs(parent)

        # Make sure that the parent is a directory and not a file
        if not os.path.isdir(parent):
            raise ValueError(
                "Please supply a directory to write preprocessed data to."
            )

        # Create a data structure for the pickle
        document = list(self.tokenize(fileid))

        # Open and serialize the pickle to disk
        with open(target, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)
            
        # Clean up the document
        del document

        # Return the target fileid
        return target

    def transform(self, fileids=None, categories=None):
        """
        Transform the wrapped corpus, writing out the segmented, tokenized,
        and part of speech tagged corpus as a pickle to the target directory.
        This method will also directly copy files that are in the corpus.root
        directory that are not matched by the corpus.fileids().
        """
        
        # Make the target directory if it doesn't already exist
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        # Resolve the fileids to start processing and return the list of 
        # target file ids to pass to downstream transformers. 
        return [
            self.process(fileid)
            for fileid in self.fileids(fileids, categories)
        ]
