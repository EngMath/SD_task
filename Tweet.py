from Linguistic_resource_AFINN import AFINN
from Linguistic_resource_DAL import DAL
from Linguistic_resource_HL import HL
from Linguistic_resource_GI import GI
import itertools
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stop=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'himself', 'herself','etc','am','pm',\
      'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this','that', 'these', 'those', 'am', 'is', 'are',\
      'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing','a', 'an', 'the', 'and', 'or', 'because','it',\
      'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into','through', 'during', 'before', 'after', 'to', 'from', 'in', 'on', 'off', 'over', \
      'again', 'further', 'then','once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',\
      'own', 'same', 'so', 'than', 'too','just','now', 'y','ve','re','m','o','s','every','ma','from','up','down','below','above','before','after','under','too','doe','to','wa','ha']



wordnet_lemmatizer = WordNetLemmatizer()

afinn = AFINN()
dal = DAL()
hl = HL()
gi=GI()



class Tweet_create(object):

    tweet_id=''
    text=''
    target=''
    stance=0

    def __init__(self, include_dictionary, tweet_id, text, target, opiniontowards, sentiment, stance=None):

        from wordsegment import load, segment
        import wordsegment as ws
        load()

        def noslang(TEXT):
            # http://www.noslang.com/dictionary
            import re
            f = open("resources/slang.txt", mode='r')

            l = f.readlines()
            ll=[elem for elem in l if elem !='\n']
            new = [x.strip('\n') for x in ll]
            dic = {}
            for e in range(0, len(new), 2):
                new[e] = " " + new[e][:-1]
                new[e] = re.escape(new[e])
                # https://www.tutorialspoint.com/How-to-escape-all-special-characters-for-regex-in-Python
                dic[new[e]] = new[e + 1]


            for e in dic.keys():
                if re.search(e.lower(), TEXT.lower()):
                    if e != "\ \@\ ":
                        TEXT = re.sub(e, " " + dic[e] + " ", TEXT, flags=re.IGNORECASE)

            return TEXT


        def baldwin_normalization(TEXT):
            # https://github.com/gouwsmeister/TextCleanser/tree/master/data/han_dataset

            f = open('resources/Baldwin_corpus_tweet1.txt', 'r')

            lines = f.readlines()
            new = [x.strip('\n') for x in lines]
            new=[elem for elem in new if  elem !='']
            last = [x.split('\t') for x in new]

            d1 = {}
            for e in last:
                try:
                    if e[0] != e[1]:
                        d1[re.escape(e[0])] = re.escape(e[1])
                except:
                    pass

            for e in d1.keys():
                if re.search(" " + e.lower() + " ", TEXT.lower()):
                    TEXT = re.sub(" " + e + " ", " " + d1[e] + " ", TEXT, flags=re.IGNORECASE)

            f = open('resources/Baldwin_common_abbrs.csv', 'r')

            lines = f.readlines()
            new = [x.strip('\n') for x in lines]
            last = [x.split(',') for x in new]
            import re
            d2 = {}
            for e in last:
                d2[re.escape(e[0])] = e[1]

            for e in d2.keys():
                if re.search(" " + e.lower() + " ", TEXT.lower()):
                    TEXT = re.sub(" " + e + " ", " " + d2[e] + " ", TEXT, flags=re.IGNORECASE)
            return TEXT



        def abbrev(TEXT):
            f = open('dictionary.csv', 'r')
            lines = f.readlines()
            new = [x.strip('\n') for x in lines]
            last = [x.split(',') for x in new]

            d1 = {}
            for e in last:
                try:
                    if e[0] != e[1]:
                        if e[0][1]=='-':
                            d1[e[0].strip()]=e[1]
                        else:
                            d1[e[0]] = e[1]
                except:
                    pass

            TEXT = " " + TEXT + " "
            for e in d1.keys():

                if e=='US':
                    if re.search(" " + e + " ", TEXT):
                        TEXT = re.sub(" " + e + " ", " " + d1[e] + " ", TEXT)
                elif e[-1]=='-':
                    if re.search(" " + e.lower(), TEXT.lower()):
                        TEXT = re.sub(" " + e , " " + d1[e] + " ", TEXT, flags=re.IGNORECASE)
                else:
                    if re.search(" " + e.lower() + " ", TEXT.lower()):
                        TEXT = re.sub(" " + e + " ", " " + d1[e] + " ", TEXT, flags=re.IGNORECASE)
            return TEXT.strip()

        wordlistfile = 'resources/wordlist.txt'
        content = None
        with open(wordlistfile) as f:
            content = f.readlines()
        wordlist = [word.rstrip('\n') for word in content]

        def ParseTag(term):
            words = []
            tags = term[1:].split('-')
            for tag in tags:
                word = FindWord(tag.lower())
                while word != None and len(tag) > 0:
                    words += [word]
                    if len(tag) == len(word):
                        break
                    tag = tag[len(word):]
                    word = FindWord(tag.lower())
            return " ".join(words)

        def ParseSentence(sentence):
            new_sentence = ""

            terms = sentence.split(' ')
            for term in terms:
                if len(term) < 1:
                    new_sentence += ""
                elif term[0] == '#':
                    new_sentence += "#" + ParseTag(term)
                elif term[0] == '@':
                    new_sentence += "@" + ParseTag(term)

                else:
                    new_sentence += term
                new_sentence += " "

            return new_sentence

        def FindWord(token):
            i = len(token) + 1
            while i > 1:
                i -= 1
                if token[:i] in wordlist:
                    return token[:i]

            return None

        import re

        self.text_raw=text

        txt=re.sub('#SemSt', '', text, flags=re.IGNORECASE)
        txt=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' URL ', txt)
        if include_dictionary==True:
            txt = abbrev(txt)

        import ekphrasis.utils.nlp as eknlp
        txt = eknlp.unpack_contractions(txt)

        txt = re.sub('RT', 'retweet', txt, flags=re.IGNORECASE)
        txt = re.sub('re-', 're', txt, flags=re.IGNORECASE)

        if ' t ' in txt:
            txt = re.sub(' t ', ' not ', txt, flags=re.IGNORECASE)
        if "'T" in txt:
            txt = re.sub("'T", 'NOT ', txt, flags=re.IGNORECASE)
        t = ""
        for e in txt.split():
            if e != 'retweet': e = re.sub('retweet', 'rt', e)
            t += e + " "
        txt = t

        txt = re.sub(r'([,.!?-])([a-zA-Z@#])', r'\1 \2',txt)
        txt = re.sub(r'([a-zA-Z])([-,.!?])', r'\1 \2',txt)
        txt=re.sub('#',' #',txt)
        txt=re.sub('@',' @',txt)

        l = []
        t = txt.split()
        for e in t:
            if e.startswith("#") or e.startswith("@"):
                f = e[0]
            else:
                f = ""
            text = ' '.join(re.findall('[a-zA-Z](?:[a-z]{1,}|[A-Z]*)', e))
            # print(text)
            if re.findall('[a-zA-Z](?:[a-z]{1,}|[A-Z]*)', e) and f == "#":
                l.append("<hashtag> " + text + " </hashtag>")
            elif re.findall('[a-zA-Z](?:[a-z]{1,}|[A-Z]*)', e) and f == "@":
                l.append("<user> " + text + " </user>")
            elif re.findall('[a-zA-Z](?:[a-z]{1,}|[A-Z]*)', e):
                l.append(f + text)
            elif f == "#":
                l.append("<hashtag> " + e.lstrip("#") + " </hashtag>")
            elif f == "@":
                l.append("<user> " + e.lstrip("@") + " </user>")

            else:
                l.append(e)
        txt = " ".join(l)


        txt=re.sub("^\d+\s|\s\d+\s|\s\d+$|\d+\.\d+|\d+:\d+ pm ", " time ", txt)
        txt=re.sub("^\d+\s|\s\d+\s|\s\d+$|\d+\.\d+|\d+:\d+ am", " time ", txt)
        txt=re.sub('\d\d\d\d', 'time', txt)
        txt=re.sub("^\d+\s|\s\d+\s|\s\d+$|\d+\.\d+", " number ", txt)
        txt=re.sub("\$\d+\.\d+", "money", txt)#
        txt=re.sub("\d+\.\d+\$", "money", txt)
        txt=re.sub("\d+\$", "money", txt)
        txt=re.sub("\d+\.\d+%", "percent", txt)
        txt=re.sub("\d+%", "percent", txt)
        txt=re.sub("it's", "it is", txt, flags=re.IGNORECASE)
        txt=re.sub("it's", "it is", txt, flags=re.IGNORECASE)
        txt=re.sub(" it's ", " it is ", txt, flags=re.IGNORECASE)
        txt=re.sub(" he's ", " he has ", txt, flags=re.IGNORECASE)
        txt=re.sub(" she's ", " she has ", txt, flags=re.IGNORECASE)
        txt= re.sub(" cant ", " can not ", txt, flags=re.IGNORECASE)
        txt= re.sub(" dont ", " do not ", txt, flags=re.IGNORECASE)
        txt= re.sub(" don ", " do not ", txt, flags=re.IGNORECASE)
        txt= re.sub(" wont ", " will not ", txt, flags=re.IGNORECASE)
        txt= re.sub(" shant ", " shall not ", txt, flags=re.IGNORECASE)
        txt=re.sub(" mustnt ", " must not ", txt, flags=re.IGNORECASE)
        txt= re.sub(" doesnt ", " does not ", txt, flags=re.IGNORECASE)
        txt= re.sub(" couldnt ", " could not ", txt, flags=re.IGNORECASE)
        txt= re.sub(" wasnt ", " was not ", txt, flags=re.IGNORECASE)
        txt= re.sub(" werent ", " were not ", txt, flags=re.IGNORECASE)
        txt= re.sub(" wasn ", " was not ", txt, flags=re.IGNORECASE)
        txt= re.sub(" weren ", " were not ", txt, flags=re.IGNORECASE)

        txt= re.sub("'s", "", txt, flags=re.IGNORECASE)
        if include_dictionary==True:
            txt = abbrev(txt)
        #print(txt)
        string = txt
        # print(txt)
        txt = re.sub(r'([,.!?#@-])([a-zA-Z])', r'\1 \2', txt)
        txt = re.sub(r'([a-zA-Z])([-#@,.!?])', r'\1 \2', txt)
        txt = re.sub(r'(\")([a-zA-Z])', r'\1 \2', txt)
        txt = re.sub(r'([a-zA-Z])(\")', r'\1 \2', txt)
        txt = " " + txt + " "


        w = ParseSentence(txt)
        n = baldwin_normalization(w)
        m = noslang(n)

        sent = ""
        mm = m.split()
        for ee in mm:
            if ee in ['<hashtag>', 'url', '</hashtag>',"<user>","</user>"]:

                seg = [ee]
            else:
                seg = segment(ee)
            if len(seg) > 1:
                if ee[0].isupper():
                    seg[0] = seg[0].capitalize()
                sent += " ".join(seg) + " "
            else:
                sent += ee + " "
        sent = sent.strip()
        #print(sent)

        if 're tweet' in sent:
            sent = re.sub('re tweet', 'retweet', sent)

        wordnet_lemmatizer = WordNetLemmatizer()
        lemmas = []

        for token in sent.split():
            if token.isupper():
                lemmas.append(wordnet_lemmatizer.lemmatize(token.lower()).upper())
            elif token[0].isupper():
                lemmas.append(wordnet_lemmatizer.lemmatize(token.lower()).capitalize())
            else:
                lemmas.append(wordnet_lemmatizer.lemmatize(token))

        txt = " ".join(lemmas)

        if include_dictionary==True:
            txt = abbrev(txt)

        txt=re.sub(" Clit on "," Clinton ",txt)
        txt=re.sub(" clit on "," clinton ",txt,flags=re.IGNORECASE)
        txt=re.sub("hill ary "," hillary ",txt,flags=re.IGNORECASE)
        txt=re.sub("hash tag hash tag","<hashtag> </hashtag>",txt,flags=re.IGNORECASE)
        txt=re.sub("hash tag number hash tag","<hashtag> number </hashtag>",txt,flags=re.IGNORECASE)
        txt=re.sub("hash tag time hash tag","<hashtag> time </hashtag>",txt,flags=re.IGNORECASE)
        txt=re.sub("hash tag money hash tag","<hashtag> money </hashtag>",txt,flags=re.IGNORECASE)
        txt=re.sub("hash tag percent hash tag","<hashtag> percent </hashtag>",txt,flags=re.IGNORECASE)

        txt = re.sub(r'(?:\w+) - (?:\w+)', "", txt)

        txt = re.sub(r"\s{2,}", " ", txt)

        tokens = [w for w in txt.split() if w.lower() not in stop]

        self.text_hash=" ".join(tokens)
        txt=re.sub('<hashtag>','#',self.text_hash)
        txt=re.sub('</hashtag>','',txt)
        txt = re.sub('<user>', '@', txt)
        self.text = re.sub('</user>', '', txt)

        self.tokens=[elem for elem in self.text.split() if elem !='']
        self.text_nopunc=re.sub(r"\W+", ' ', self.text)
        self.tokens_nopunc= [elem for elem in self.text_nopunc.split() if elem !='']
        self.word2vecinput=[elem for elem in self.text_nopunc.lower().split() if elem !='']


        self.tweet_id = tweet_id


        self.pos = [ token[1] for token in nltk.pos_tag(self.text.split())]

        self.sentimentafinn=afinn.get_afinn_sentiment(self.text)
        self.sentimentdal=dal.get_dal_sentiment(self.text)
        self.sentimenthl=hl.get_HL_sentiment(self.text)

        self.sentimentgi=gi.get_gi(self.text)
        self.target=target
        self.stance=stance

        self.opiniontowards=opiniontowards
        self.sentiment=sentiment



def make_tweet(include_dictionary,tweet_id, text, target, labeledopiniontowards,labeledsentiment, stance=None):

    tweet = Tweet_create( include_dictionary,tweet_id, text, target, labeledopiniontowards,labeledsentiment, stance=stance)

    return tweet