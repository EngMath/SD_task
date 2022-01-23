
import codecs,subprocess,tempfile
from collections import Counter, OrderedDict

import numpy as np
import nltk
from twitterTokenizer import Tokenizer


negation = set(["never","no","nothing","nowhere","noone","none","not","havent","haven't","hasnt","hasn't","hadnt","hadn't", 
                "cant","can't","couldnt","couldn't","shouldnt","shouldn't","wont","won't","wouldnt","wouldn't","dont","don't","doesnt","doesn't","didnt",
                "didn't","isnt","isn't","arent","aren't","aint","ain't"])
    
def get_pos_tags(tweetText):

    tf = tempfile.NamedTemporaryFile(delete=False)
    with codecs.open(tf.name, 'w', encoding='utf8') as out:
        for i in tweetText:
            out.write("%s\n" % i)
    #com = "C:/MinGW/msys/1.0/bin/sh runTagger.sh %s"%tf.name
    com = "C:/MinGW/msys/1.0/bin/sh twise/runTagger.sh %s"%tf.name

    op= subprocess.check_output(com.split())
    op = op.splitlines()

    pos_text = [str(x).split("\t")[0].split() for x in op]
    pos_text = [str(x).split(r"\t")[0].split() for x in op]

    pos = [str(x).split(r"\t")[1].split() for x in op]

    different_pos_tags = list(set([x for i in pos for x in i]))
    pos_features = []
    for instance in pos:
        tags = []
        instance = Counter(instance)#https://pymotw.com/2/collections/counter.html
        for pos_tag in different_pos_tags:
            try:
                tags.append(instance[pos_tag])
            except:
                tags.append(0)
        pos_features.append(np.array(tags))
    pos_features = np.array(pos_features)
    #print "------------\nPOS-tagging finished!\n------------\nThere are %d pos-tags (incl. hashtags). Shape: %d,%d"%(len(different_pos_tags), pos_features.shape[0],  pos_features.shape[1])
    for key1, i in enumerate(pos_text):# key1 is the index of whole tweet , i is the whole tweet text
        flag = False
        for key, j in enumerate(i):#key is the word index in the tweet , j is the word itself
            i[key] = j.lower()
            if flag:# we add negation if the word is AVRN and it follow a negation .. until we find a word that is not AVRN
                if pos[key1][key] in "AVRN" :
                    i[key]+="_NEG"
                else:
                    flag=False
            if j in negation:
                flag = True
    return pos, pos_features, different_pos_tags, pos_text

        
def mpqa(tweetText, pos, different_pos_tags, pos_text,tknzr):
    #voca = codecs.open('lexicons/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff', 'r').read().splitlines()
    voca = codecs.open('resources/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff', 'r').read().splitlines()

    wds1, wds = {}, {}
    for i in voca:
        i = i.split()
        try:
            if wds1[i[2].split('=')[1]] != i[5].split('=')[1]:
                pass
        except:
            if i[5].split('=')[1] in ['positive', 'negative']:
                wds1[i[2].split('=')[1]] = i[5].split('=')[1]
                wds[i[2].split('=')[1]]=(i[0].split('=')[1], i[5].split('=')[1])
    feat = []
    if tknzr=="twise":
        tokenizer = Tokenizer(preserve_case=False).tokenize
    elif tknzr=="nltk":
        tokenizer=nltk.word_tokenize
    for key, tweet in enumerate(pos_text):
        direction = {'negative':-1, 'positive':1, 'neutral':0, 'both':0, 'weaksubj':1, 'strongsubj':2}
        pp, pn, npp, nn, pp_hash, pn_hash, npp_hash, nn_hash  = 0,0,0,0,0,0,0,0
        words=tokenizer(tweet)
        for i in words:
            if i in wds:
                if direction[wds[i][1]] > 0:
                    pp += direction[wds[i][0]]*direction[wds[i][1]]
                if direction[wds[i][1]] < 0:#negative
                    pn += direction[wds[i][0]]*direction[wds[i][1]]
            if i.endswith("_neg"):
                my_i = i.strip("_neg")
                if my_i in wds:
                    if direction[wds[my_i][1]] > 0:
                        npp += direction[wds[my_i][0]]*direction[wds[my_i][1]]
                    if direction[wds[my_i][1]] < 0:
                        nn += direction[wds[my_i][0]]*direction[wds[my_i][1]]
            if i[0] == "#":
                if i[1:] in wds:
                    if direction[wds[i[1:]][1]] > 0:
                        pp_hash += direction[wds[i[1:]][0]]*direction[wds[i[1:]][1]]
                    if direction[wds[i[1:]][1]] < 0:
                        pn_hash += direction[wds[i[1:]][0]]*direction[wds[i[1:]][1]]
                if i.endswith("_neg"):
                    my_i = i[1:].strip("_neg")
                    if my_i in wds:
                        if direction[wds[my_i][1]] > 0:
                            npp_hash += direction[wds[my_i][0]]*direction[wds[my_i][1]]
                        if direction[wds[my_i][1]] < 0:
                            nn_hash += direction[wds[my_i][0]]*direction[wds[my_i][1]]
        pos_sen = OrderedDict({x:[0,0,0,0] for x in different_pos_tags})
        for k_key, i in enumerate(pos_text[key]):
            if i in wds:
                if direction[wds[i][1]] > 0:
                    pos_sen[pos[key][k_key]][0]+=1
                if direction[wds[i][1]] < 0:
                    pos_sen[pos[key][k_key]][1]+=1
            if i.endswith("_NEG"):
                if i.strip('_NEG') in wds:
                    ii = i.strip('_NEG')
                    if direction[wds[ii][1]] > 0:
                        pos_sen[pos[key][k_key]][2]+=1
                    if direction[wds[ii][1]] < 0:
                        pos_sen[pos[key][k_key]][3]+=1
        my_feat = [pp, pn, npp, nn]+[g for gg in pos_sen.values() for g in gg]
        feat.append(np.array(my_feat))
    return np.array(feat)

def get_sentiwordnet(pos_text, pos):
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.corpus import sentiwordnet as swn
    feat = []
    for key, val in enumerate(pos):
        pos, neg, pos_neg, neg_neg, POS_pos, POS_neg = 0,0,0,0, {'A':0, 'V':0, 'R':0, 'N':0}, {'A':0, 'V':0, 'R':0, 'N':0}
        for key3, val3 in enumerate(val):
            if val3 in 'AVRN':
                text = pos_text[key][key3].strip("_NEG")#
                synsets = wn.synsets('%s'%text,val3.lower())

                try:
                    sense=synsets[0] #
                except:
                    continue
                k = swn.senti_synset(sense.name())
                if k != None:
                    if pos_text[key][key3].endswith("_NEG"):
                        pos_neg += k.pos_score()
                        neg_neg += k.neg_score()
                        POS_neg[val3]+=1
                    else:
                        pos += k.pos_score()
                        neg += k.neg_score()
                        POS_pos[val3]+=1
        feat.append([pos, neg, pos_neg, neg_neg, pos+neg+pos_neg+neg_neg, sum(POS_pos.values())+sum(POS_neg.values())]+list(POS_pos.values())+list(POS_neg.values()))
    return np.array(feat)


def bing_lius(tweetText, pos, different_pos_tags, pos_text,tknzr ):
    #with codecs.open('lexicons/positive-words_bing_liu.txt', 'r') as inFile:
    with codecs.open('resources/positive-words_bing_liu.txt', 'r') as inFile:
        positive = set(inFile.read().splitlines())
    #with codecs.open('lexicons/negative-words_bing_liu.txt', 'r') as inFile:
    with codecs.open('resources/negative-words_bing_liu.txt', 'r') as inFile:
        negative = set(inFile.read().splitlines())
    feat = []
    if tknzr=="twise":
        tokenizer = Tokenizer(preserve_case=True).tokenize
    elif tknzr=="nltk":
        tokenizer=nltk.word_tokenize
    
    for key, tweet in enumerate(pos_text):
        words=tokenizer(tweet)
        counters, counters_cap = np.zeros(4), np.zeros(4)
        for j in words:
            if j.isupper():
                counters_cap += np.array(getBingLiusCounters(positive, negative, j.lower()))

            else:
                counters += np.array(getBingLiusCounters(positive, negative, j.lower()))
        pos_sen = OrderedDict({x:[0,0,0,0] for x in different_pos_tags})
        for k_key, k in enumerate(pos_text[key]):
            if k in positive:
                pos_sen[pos[key][k_key]][0]+=1
            if k in negative:
                pos_sen[pos[key][k_key]][2]+=1
            if k.endswith("_NEG"):
                if k.strip("_NEG") in positive:
                    pos_sen[pos[key][k_key]][1]+=1
                if k.strip("_NEG") in negative:
                    pos_sen[pos[key][k_key]][3]+=1
        my_feat = list(counters+counters_cap)+[g for gg in pos_sen.values() for g in gg]
        feat.append(np.array(my_feat))
    return np.array(feat)


def getBingLiusCounters(positive, negative, i):
    pp, pn, npp, nn, pp_hash, pn_hash, npp_hash, nn_hash = 0,0,0,0,0,0,0,0
    if i in positive:
        pp+=1
    if i in negative:
        npp+=1
    if i.endswith("_neg"):
        if i.strip("_neg") in positive:
            pn+=1
        if i.strip("_neg") in negative:
            nn+=1
    if i[0] == "#":
        if i[1:] in positive:
            pp_hash+=1
        if i[1:] in negative:
            npp_hash+=1
        if i.endswith("_neg"):
            if i[1:].strip("_neg") in positive:
                pn_hash+=1
            if i[1:].strip("_neg") in negative:
                nn_hash+=1
    return pp, pn, npp, nn


def nrc_emotion(tweetText, pos, different_pos_tags, pos_text ,tknzr):
    #Each line has the following format: TargetWord<tab>AffectCategory<tab>AssociationFlag
    #TargetWord is a word for which emotion associations are provided.
    #AffectCategory is one of eight emotions (anger, fear, anticipation, trust, surprise, sadness, joy, or disgust) or one of two polarities (negative or positive).
    #AssociationFlag has one of two possible values: 0 or 1.  0 indicates that the target word has no association with affect category, whereas 1 indicates an association

    #with codecs.open('lexicons/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt', 'r') as inFile:
    with codecs.open('resources/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt', 'r') as inFile:
        wds = inFile.read().splitlines()
    positive, negative = [], []
    for i in wds:
        my_i = i.split('\t')
        if my_i[1] == 'positive' and my_i[2]=='1':
            positive.append(my_i[0])
        if my_i[1] == 'negative' and my_i[2]=='1':
            negative.append(my_i[0])
    feat = []
    positive, negative = set(positive), set(negative)
    if tknzr=="twise":
        tokenizer=Tokenizer(preserve_case=True).tokenize
    elif tknzr=="nltk":
        tokenizer=nltk.word_tokenize
        
    for key, tweet in enumerate(tweetText):
        words=tokenizer(tweet)
        counters, counters_caps = np.zeros(8), np.zeros(8)
        for i in words:
            if i.isupper():
                counters_caps += np.array(getBingLiusCounters(positive, negative, i.lower()))
            else:
                counters += np.array(getBingLiusCounters(positive, negative, i.lower()))
        pos_sen = OrderedDict({x:[0,0,0,0] for x in different_pos_tags})
        for k_key, k in enumerate(pos_text[key]):
            if k in positive:
                pos_sen[pos[key][k_key]][0]+=1
            if k in negative:
                pos_sen[pos[key][k_key]][2]+=1
            if k.endswith("_NEG"):
                if k.strip("_NEG") in positive:
                    pos_sen[pos[key][k_key]][1]+=1
                if k.strip("_NEG") in negative:
                    pos_sen[pos[key][k_key]][3]+=1
#        my_feat = list(counters)+list(counters_caps)+[g for gg in pos_sen.values() for g in gg]
        my_feat = list(counters+counters_caps)+[g for gg in pos_sen.values() for g in gg]
        feat.append(np.array(my_feat))
    return np.array(feat)


def sent140aff(tweetText, pos, different_pos_tags, pos_text, path2lexicon,tknzr):
    #'../lexicons/HashtagSentimentAffLexNegLex/HS-AFFLEX-NEGLEX-unigrams.txt'
    #'../resources/HashtagSentimentAffLexNegLex/HS-AFFLEX-NEGLEX-unigrams.txt'
    #Each line in the lexicons has the following format: <term><tab><score><tab><Npos><tab><Nneg>
    #<term> can be a unigram or a bigram;
    #<score> is a real-valued sentiment score: score = PMI(w, pos) - PMI(w, neg), where PMI stands for Point-wise Mutual Information between a term w and the positive/negative class;
    #<Npos> is the number of times the term appears in the positive class, ie. in tweets with positive hashtag or emoticon;
    #<Nneg> is the number of times the term appears in the negative class, ie. in tweets with negative hashtag or emoticon.
##    Both parts, AffLex and NegLex, of each lexicon are contained in the same file. The NegLex entries have suffixes '_NEG' or '_NEGFIRST
##    In the unigram lexicon:
##'_NEGFIRST' is attached to terms that directly follow a negator;
##'_NEG' is attached to all other terms in negated contexts (not directly following a negator).
##In the bigram lexicon:
##'_NEG' is attached to all terms in negated contexts.
##Both suffixes are attached only to nouns, verbs, adjectives, and adverbs. All other parts of speech do not get these suffixes attached. 

    with codecs.open(path2lexicon, 'r') as inFile:
        wds = inFile.read().splitlines()
    pos_cont, nega_cont, nega_cont_first = {},{},{}
    for i in wds:
        i = i.split("\t")
        if i[0].endswith("_NEG"):
            name = "".join(i[0].split('_')[:-1])
            nega_cont[name]=float(i[1])
        elif i[0].endswith('_NEGFIRST'):
            name = "".join(i[0].split('_')[:-1])
            nega_cont_first[name]=float(i[1])
        else:
            pos_cont[i[0]]=float(i[1])
    feat = []
    if tknzr=="twise":
        tokenizer = Tokenizer(preserve_case=False).tokenize
    elif tknzr=="nltk":
        tokenizer = nltk.word_tokenize
    for key, tweet in enumerate(pos_text):
        cnt, scor  = 0, []
        words=tokenizer(tweet)
        for my_key, i in enumerate(words):
            if i in pos_cont:
                scor.append(pos_cont[i])
            if i.endswith('_neg'):
                j = i.strip("_neg")
                flag = 0
                if not words[my_key-1].endswith('_neg'):#
                    if j in nega_cont_first:
                        scor.append(nega_cont_first[j])
                        flag = 1
                    elif j in nega_cont:
                        scor.append(nega_cont[j])
                        flag = 1 
                    else:
                        pass
                if j in nega_cont and flag == 0:

                    scor.append(nega_cont[j])
        if len(scor)> 0:
            pos_scores, neg_scores = [x for x in scor if x>0],[x for x in scor if x<0]
            if len(pos_scores) == 0:
                pos_scores= [0]
            if len(neg_scores) == 0:
                neg_scores=[0]
            feat.append([len(scor), len(pos_scores), len(neg_scores), sum(scor), sum(pos_scores), sum(neg_scores), max(scor), 
                        max(pos_scores), max(neg_scores), scor[-1], pos_scores[-1], neg_scores[-1]])#
        else:
            feat.append(list(np.zeros(12)))
    return np.array(feat)

def sent140aff_bigrams(tweetText, pos, different_pos_tags, pos_text, path2lexicon):
    with codecs.open(path2lexicon, 'r') as inFile:
        wds = inFile.read().splitlines()
    lexicon = {}
    for i in wds:
        i = i.split("\t")
        lexicon[i[0]]=float(i[1])
    feat = []
    for key,tweet in enumerate(pos_text):
        scor = []

        bigrams = zip(tweet, tweet[1:])
        for pair in bigrams:
            look = " ".join(pair)
            if look in lexicon:
                scor.append(lexicon[look])
        if len(scor)> 0:
            pos_scores, neg_scores = [x for x in scor if x>0],[x for x in scor if x<0]
            if len(pos_scores) == 0:
                pos_scores= [0]
            if len(neg_scores) == 0:
                neg_scores=[0]
            feat.append([len(scor), len(pos_scores), len(neg_scores), sum(scor), sum(pos_scores), sum(neg_scores), max(scor),
                        max(pos_scores), max(neg_scores), scor[-1], pos_scores[-1], neg_scores[-1]])
        else:
            feat.append(list(np.zeros(12)))
    return np.array(feat)

