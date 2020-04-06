import os
import logging

import nltk
from nltk import sent_tokenize
import spacy
from nltk.tag.stanford import StanfordPOSTagger
from stanfordnlp.server import CoreNLPClient

# Logging channel
logger = logging.getLogger(__name__)

# Universal Part of Seech Tagger
UTagSet = {
    'ADJ'   :   'adjective',	
    'ADP'   :   'adposition',	
    'ADV'   :   'adverb',	
    'CONJ'  :   'conjunction',	
    'DET'   :   'determiner, article',	
    'NOUN'  :   'noun',	
    'NUM'   :   'numeral',	
    'PRT'   :   'particle',	
    'PRON'  :   'pronoun',	
    'VERB'  :   'verb',	
    '.'     :   'punctuation marks',
    'X'	:   'other',
}

NLTKTagSet = {
    'CC'    :   'coordinating conjunction',
    'CD'    :   'cardinal digit',
    'DT'    :   'determiner',
    'EX'    :   'existential there',
    'FW'    :   'foreign word',
    'IN'    :   'preposition/subordinating conjunction',
    'JJ'    :   'adjective',
    'JJR'   :   'adjective, comparative',
    'JJS'   :   'adjective, superlative',
    'LS'    :   'list marker',
    'MD'    :   'modal',
    'NN'    :   'noun, singular',
    'NNS'   :   'noun plural',
    'NNP'   :   'proper noun, singular',
    'NNPS'  :   'proper noun, plural',
    'PDT'   :   'predeterminer',
    'POS'   :   'possessive ending',
    'PRP'   :   'personal pronoun',
    'PRP$'  :   'possessive pronoun',
    'RB'    :   'adverb',
    'RBR'   :   'adverb, comparative',
    'RBS'   :   'adverb, superlative',
    'RP'    :   'particle',
    'TO'    :   'to',
    'UH'    :   'interjection',
    'VB'    :   'verb, base form',
    'VBD'   :   'verb, past tense',
    'VBG'   :   'verb, gerund/present participle',
    'VBN'   :   'verb, past participle',
    'VBP'   :   'verb, sing. present, non-3d',
    'VBZ'   :   'verb, 3rd person sing. present',
    'WDT'   :   'wh-determiner',
    'WP'    :   'wh-pronoun',
    'WP$'   :   'possessive wh-pronoun',
    'WRB'   :   'wh-abverb',
}

# Universal Dependencies TAG Set
UDTagSet = {
    'ADJ'   :	'adjective',
    'ADP'   :	'adposition',
    'ADV'   :	'adverb',
    'AUX'   :	'auxiliary',
    'CONJ'  :	'conjunction',
    'CCONJ' :	'coordinating conjunction',	
    'DET'   :	'determiner',
    'INTJ'  :	'interjection',
    'NOUN'  :	'noun', 	
    'NUM'   :	'numeral',	
    'PART'  :	'particle',
    'PRON'  :	'pronoun',	
    'PROPN' :	'proper noun',	
    'PUNCT' :	'punctuation',	
    'SCONJ' :	'subordinating conjunction',
    'SYM'   :	'symbol',
    'VERB'  :	'verb',	
    'X'	    :   'other',
    'SPACE' :	'space',
}

# French Treebank TAG Set
FTTagSet = {
    'ADJ'   :   'adjective',
    'ADJWH' :   'interrogative adjective',
    'ADV'   :   'adverb',
    'ADVWH' :	'interrogative adverb',
    'CC'    :   'coordinating conjunction',
    'Cl'    :   'weak clitic pronoun',
    'CS'    :   'subordinating conjunction',
    'DET'   :   'determiner',
    'ET'    :   'foreign word',
    'I'     :   'interjection',
    'NC'    :   'common noun',
    'NPP'   :   'proper noun',
    'P'     :   'preposition',
    'PREF'  :   'prefix',
    'PRO'   :   'full pronoun',
    'PROWH' :	'interrogative pronoun',
    'V'     :   'verb',
    'VIMP'  :   'imperative verb form',
    'VINF'  :   'infinitive verb form',
    'VPP'   :   'past participle',
    'VPR'   :   'present participle',
    'VS'    :   'subjunctive verb form',
    'PUNC'  :   'punctuation mark',
    'N'     :   'noun',
    'PROREL':   'relative pronoun',
}

class pos_tagger() :
    """
    Class to impement part of speech tagging (pos tagging)
    """

    
    def __init__(self, tagger='spacy', language='french') :
        self.tagger = tagger
        self.tagmodule = None
        self.tagset = UTagSet              # TAG Set by default
        self.language = language
        spacy_module = { 'french' : 'fr_core_news_sm',
                         'english': 'en_core_web_sm' }

        if tagger == 'spacy' :
            self.tagger = self.spacy_pos_tag
            self.tagset = UDTagSet
            try :
                self.tagmodule = spacy.load(spacy_module[language])
            except :
                logger.warning('Module for language [{:s}] not installed for Spacy - using french by default'.format(language))
                self.tagmodule = spacy.load(spacy_module['french'])
        elif tagger == 'stanford' :
            self.tagger = self.stanford_pos_tag
            self.tagset = FTTagSet
            JAVAHOME = "C:/Program Files (x86)/Java/jre1.8.0_241/bin/java.exe"
            # Set a JAVAHOME environment variable if not present
            if not 'JAVAHOME' in os.environ : 
                os.environ['JAVAHOME'] = JAVAHOME
            root_path="./stanford-postagger/" # location of Stanford POS Tagger components
    
            # Launch the Stanford Pos Tagger (implemented in Java)
            self.tagmodule = StanfordPOSTagger(root_path + "models/"+language+".tagger",
                                               root_path + "stanford-postagger.jar",encoding='utf8')
        elif tagger == 'core_nlp' :
            self.tagger = self.corenlp_pos_tag
            os.environ['CORENLP_HOME'] = './stanford-corenlp-full-2018-10-05'
            try : 
                self.tagmodule = CoreNLPClient(properties=language, annotators=['pos', ], timeout=30000, memory='1G')
            except :
                logger.warning('Could not launch Stanford Core NLP for [{:s}]'.format(language))
        elif tagger == 'nltk' :
            self.tagger = self.nltk_pos_tag
            self.tagset = NLTKTagSet
            if language != 'english' :
                logger.warning('nltk does not support [{:s}] language'.format(language))
        else :
            logger.warning('POS tagger [{:s}] unknown'.format(tagger))
            
    def pos_tag(self, sentence) :
        assert(self.tagger)
        return self.tagger(sentence)

    def spacy_pos_tag(self, sentence) :
        assert(self.tagmodule)
        return [(token.text, token.pos_) for token in self.tagmodule(sentence)]

    def stanford_pos_tag(self, sentence) :
        assert(self.tagmodule)
        return self.tagmodule.tag(nltk.word_tokenize(sentence))
    
    def corenlp_pos_tag(self, sentence) :
        # Unchecked
        # DOes not seem to work 
        assert(self.tagmodule)
        ann = self.tagmodule.annotate(sentence)
        return [(token.word, token.pos) for token in ann.sentence[0].token ]

    def nltk_pos_tag(self, sentence) :
        return nltk.pos_tag(nltk.word_tokenize(sentence))

    def tag_label(self, tag) :
        return self.tagset.get(tag, '??')


if __name__ == '__main__':

    sents = [   "Je suis un artiste peintre qui s'ignore!",
                "Renoir peignait dans son atelier des nus",
                "Membre du groupe De Stijl, Piet Mondrian est principalement connu pour ses peintures abstraites aux lignes épurées et ses carrés rouge, jaune et bleu.",
                "Le musée Marmottan Monet lui consacre une exposition événement en septembre 2019 et met l’accent sur son œuvre figurative majeure.",
                "Une soixantaine de peintures de premier ordre, sélectionnées par Mondrian lui-même vers 1920 pour son plus grand collectionneur Salomon B. Slijper, sont présentées en exclusivité à Paris et révèlent cette face méconnue de l’artiste.",
                "Paysages, portraits, peintures de fleurs marquées par l’impressionnisme, le luminisme, les fauves et le symbolisme font face à de rares compositions cubistes et néo-plasticistes et placent l’artiste au rang des premiers coloristes de son temps et des grands maitres de la peinture figurative du XXe siècle.",
                "Une invitation à découvrir un autre Mondrian"
            ]
    sents = [ 'Van Eyck. Une Révolution optique.',
              'Seule une vingtaine d’œuvres du Maître flamand Van Eyck sont conservées de par le monde. Tout à fait exceptionnellement, au moins la moitié d’entre elles feront le voyage jusqu’à Gand en 2020, pour l’exposition « Van Eyck. Une Révolution optique » au Musée des Beaux-Arts de Gand (MSK Gent). Un évènement incontournable, véritable tour de force qui rendra l’univers de Van Eyck et son regard révolutionnaire plus tangibles que jamais.',
              'L’exposition s’articule autour des volets extérieurs restaurés de «L’Adoration de l’Agneau mystique» et d’autres œuvres de Van Eyck.',
              'Pour que la révolution optique de Van Eyck soit clairement perceptible, ses tableaux seront installés à côté d’œuvres de ses contemporains les plus talentueux originaires d’Allemagne, d’Espagne, de France, et d’Italie. « Van Eyck. Une Révolution optique » sera une expérience unique à vivre au cours de l’année 2020 que Gand consacre à Van Eyck.',
    ]

    for tagger in [ 'nltk', 'spacy', 'stanford' ] :
        print('===== '+tagger+' =====')
        pt = pos_tagger(tagger, language='french')
        for sentence in sents : 
            tags = pt.pos_tag(sentence)
            for tag in tags :
                word, wtype = tag
                if True : #wtype in [ 'NPP', ]:
                    print('{:40.40s} ({:s}-{:s})'.format(word, wtype, pt.tag_label(wtype)))
