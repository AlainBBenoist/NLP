import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as tts

class CorpusLoader(object):

    def __init__(self, reader, folds=12, shuffle=True, categories=None):
        self.reader = reader
        self.folds  = KFold(n_splits=folds, shuffle=shuffle)
        self.files  = np.asarray(self.reader.fileids(categories=categories))

    def fileids(self, idx=None):
        if idx is None:
            return self.files
        return self.files[idx]

    def documents(self, idx=None):
        for fileid in self.fileids(idx):
            yield list(self.reader.docs(fileids=[fileid]))

    def labels(self, idx=None):
        return [
            self.reader.categories(fileids=[fileid])[0]
            for fileid in self.fileids(idx)
        ]

    def __iter__(self):
        for train_index, test_index in self.folds.split(self.files):
            X_train = self.documents(train_index)
            y_train = self.labels(train_index)

            X_test = self.documents(test_index)
            y_test = self.labels(test_index)

            yield X_train, X_test, y_train, y_test


if __name__ == '__main__':
    from reader import PickledCorpusReader

    corpus = PickledCorpusReader('../pickle_corpus')
    loader = CorpusLoader(corpus, 12, categories=["expositions", "expositions_galeries", 'marches_salons'])
    fcount = dcount = pcount = scount= tcount=0
    for x_train, x_test, y_train, x_test in loader :
        for file in x_train :
            fcount += 1
            for doc in file :
                dcount += 1
                for para in doc :
                    #print(para)
                    #key=input('>>')
                    pcount += 1
                    #if pcount >= 1 :
                    #    continue
                    for sent in para :
                        scount += 1
                        for tag in sent :
                            #print(tag)
                            tcount+=1
    print('{:d} files,  {:d} docs, {:d} paras, {:d} sents, {:d} words found'.format(fcount, dcount, pcount, scount, tcount))
