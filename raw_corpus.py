#!/usr/bin/env python
# coding: utf-8
# TO DO:

import os
import sys
import logging
import codecs
import hashlib

# Logging channel
logger = logging.getLogger(__name__)

class corpus_builder() :
    """
    Class used to process and save events in a raw corpus
    """
    encoding='utf-8'
    extension='.json'
    
    def __init__(self, corpus_dir) :
        """
        Initialization of corpus_builder
        """
        self.corpus_dir = corpus_dir
        self.counter = dict()

    def save_file(self, path, filename, title, content) :
        """
        Save a content to the corpus directory
        """
        if os.path.exists(path) is False :
            # Create the directory
            try :
                os.mkdir(path)
            except :
                logger.warning('could not create {:s}'.format(path))
                return False
        # Save the file
        with codecs.open(filename, 'w', encoding=self.encoding) as f:
            full_text='<h1>'+title+'</h1>\n'+content
            f.write(full_text)
        f.close()
        return True

    def process(self, item, category, filehex=None) :
        """
        Process an event or a venue and record statistics
        """
        # If filename no specified, build one from the url 
        if not filehex :
            # Get URL of event 
            url = item['slug']
        
            # Encode file name
            filehash = hashlib.md5(url.encode())
            filehex = filehash.hexdigest()
        
        # Get Title
        title = item['name']
        logger.info('{:40.40s}\t{:32.32s}\t{:s}{:s}'.format(title, category, filehex, self.extension)) 

        # Get Description
        if 'description' in item :
            description = item['description']
        else :
            description = ''

        # Save the content to the corpus directory
        path = self.corpus_dir + '/' + category.replace('-', '_').lower().strip()
        filename = path+'/'+filehex+self.extension
        self.save_file(path, filename, title, description)

        # Keep track of statistics    
        if category in self.counter :
            self.counter[category]+= 1
        else :
            self.counter[category]=1

    def get_stats(self) :
        """
        Return statistics
        """
        return self.counter

# Body of program
if __name__ == '__main__':
    event = {'slug' : 'https://dubutade.fr',
             'name' : 'Exposition de peinture',
             'description' : '<p>texte de description</p>',
             }
    corp = corpus_builder('./')
    corp.process(event, 'corpus', filehex='toto')
    corp.get_stats()
