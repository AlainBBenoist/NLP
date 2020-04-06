import os
from reader import HTMLCorpusReader, PickledCorpusReader
from preprocess import Preprocessor

if __name__ == '__main__':
    fileids = ['demos_artistes/298f0f25af2ed7310271463dfd655dc9.json',]
    fileids = None
    corpus = HTMLCorpusReader('./raw_corpus')
    stats = corpus.describe()
    print('Corpus: {:d} files, {:d} paragraphs, {:d} sentences, {:d} words'.format(stats['files'], stats['paras'], stats['sents'], stats['words']))

    # Create a Pickle Corpus
    pp = Preprocessor(corpus, './pickle_corpus')
    pp.transform(fileids=fileids)



