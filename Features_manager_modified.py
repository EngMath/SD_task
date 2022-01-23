import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy import sparse
import pickle
import nltk
from nltk.corpus import stopwords
import os.path
import warnings 
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim') 
from gensim.models import word2vec

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel
import re


import codecs
import gensim
from sklearn.externals import joblib



class Features_manager(object):
    def __init__(self,TARGET,op):
        self.TARGET=TARGET
        if TARGET == "Atheism":
            self.pathfile = r"atheism\\"+"op"+str(op)+r"\\"
        elif TARGET == "Feminist Movement":
            self.pathfile = r"feminist\\"+"op"+str(op)+r"\\"
        elif TARGET == "Legalization of Abortion":
            self.pathfile = r"abortion\\"+"op"+str(op)+r"\\"
        elif TARGET == "Climate Change is a Real Concern":
            self.pathfile = r"climate\\"+"op"+str(op)+r"\\"
        elif TARGET == "Hillary Clinton":
            self.pathfile = r"hc\\"+"op"+str(op)+r"\\"

        return
    def get_stance(self,tweets):
        stance=[]
        for tweet in tweets:
            stance.append(tweet.stance)
        return stance


    def get_sentiment(self,tweets,tweets_train):
        if os.path.isfile(self.pathfile+"train_"+"get_sentiment"+'.pkl'):
            sentiment_train=joblib.load(self.pathfile+"train_"+"get_sentiment"+'.pkl')
            if os.path.isfile(self.pathfile+"test_"+"get_sentiment"+'.pkl'):
                sentiment=joblib.load(self.pathfile+"test_"+"get_sentiment"+'.pkl')
                return csr_matrix(np.vstack(sentiment_train)), csr_matrix(np.vstack(sentiment)), ["feature_sentiment"]


        sentiment=[]
        sentiment_train=[]
        for tweet in tweets:
            sentiment.append(tweet.sentiment)
        for tweet in tweets_train:
            sentiment_train.append(tweet.sentiment)
        joblib.dump(sentiment_train,self.pathfile+"train_"+"get_sentiment"+'.pkl',protocol=2)
        joblib.dump(sentiment,self.pathfile+"test_"+"get_sentiment"+'.pkl',protocol=2)

        return csr_matrix(np.vstack(sentiment_train)),csr_matrix(np.vstack(sentiment)),["feature_sentiment"]
    
    def get_opinion(self,tweets,tweets_train):
        if os.path.isfile(self.pathfile+"train_"+"get_opinion"+'.pkl'):
            opinion_train=joblib.load(self.pathfile+"train_"+"get_opinion"+'.pkl')
            if os.path.isfile(self.pathfile+"test_"+"get_opinion"+'.pkl'):
                opinion=joblib.load(self.pathfile+"test_"+"get_opinion"+'.pkl')
                return csr_matrix(np.vstack(opinion_train)), csr_matrix(np.vstack(opinion)), ["feature_opinion"]


        opinion=[]
        opinion_train=[]
        for tweet in tweets:
            opinion.append(tweet.opiniontowards)
        for tweet in tweets_train:
            opinion_train.append(tweet.opiniontowards)

        joblib.dump(opinion_train,self.pathfile+"train_"+"get_opinion"+'.pkl',protocol=2)
        joblib.dump(opinion,self.pathfile+"test_"+"get_opinion"+'.pkl',protocol=2)
        return csr_matrix(np.vstack(opinion_train)),csr_matrix(np.vstack(opinion)),["feature_opinion"]

    def tweet_len(self,tweets,tweets_train):#uwb paper

        trainfile=self.pathfile+"train_"+"tweet_len"+'.pkl'
        testfile=self.pathfile+"test_"+"tweet_len"+'.pkl'

        if os.path.isfile(trainfile):
            train=joblib.load(trainfile)
            if os.path.isfile(testfile):
                test=joblib.load(testfile)
                return csr_matrix(train).T, csr_matrix(test).T, ["tweet_len"]

        train=[]
        test=[]
        for tweet in tweets:
            test.append(len(nltk.word_tokenize(tweet.text_nopunc)))
        for tweet in tweets_train:
            train.append(len(nltk.word_tokenize(tweet.text_nopunc)))
        joblib.dump(train,trainfile,protocol=2)
        joblib.dump(test,testfile,protocol=2)
        return csr_matrix(train).T,csr_matrix(test).T,["tweet_len"]
    
    def avg_word_len(self,tweets,tweets_train):#takelab paper

        trainfile = self.pathfile+"train_" + "avg_word_len" + '.pkl'
        testfile = self.pathfile+"test_" + "avg_word_len" + '.pkl'

        if os.path.isfile(trainfile):
            avg_list_train = joblib.load(trainfile)
            if os.path.isfile(testfile):
                avg_list_test = joblib.load(testfile)
                return csr_matrix(avg_list_train).T, csr_matrix(avg_list_test).T, ["avg_word_len"]

        avg_list_train=[]
        for tweet in tweets_train:
            avg=[]
            for word in nltk.word_tokenize(tweet.text_nopunc):
                avg.append(len(word))
            avg=np.mean(np.array(avg))
            avg_list_train.append(avg)

        avg_list_test=[]
        for tweet in tweets:
            avg=[]
            for word in nltk.word_tokenize(tweet.text_nopunc):
                avg.append(len(word))
            avg=np.mean(np.array(avg))
            avg_list_test.append(avg)
        joblib.dump(avg_list_train,trainfile,protocol=2)
        joblib.dump(avg_list_test,testfile,protocol=2)
        return csr_matrix(avg_list_train).T,csr_matrix(avg_list_test).T,["avg_word_len"]

    def seq_vowel(self,tweets,tweets_train):#takelab paper
        #https://stackoverflow.com/questions/6080008/regex-in-python-to-find-words-that-follow-pattern-vowel-consonant-vowel-cons
        trainfile = self.pathfile+"train_" + "seq_vowel" + '.pkl'
        testfile = self.pathfile+"test_" + "seq_vowel" + '.pkl'

        if os.path.isfile(trainfile):
            avg_list_train = joblib.load(trainfile)
            if os.path.isfile(testfile):
                avg_list_test = joblib.load(testfile)
                return csr_matrix(avg_list_train), csr_matrix(avg_list_test), ['len_vowelseq', 'bool_vowelseq']


        train_len = [len(re.findall(r"(a{2,}|e{3,}|i{2,}|o{3,}|u{2,})", tweet.text)) for tweet in
                     tweets_train]  # length of seq of vowel in each tweet
        ##for example : re.findall(r"(a{2,}|e{2,}|i{2,}|o{2,}|u{2,})",'goooeoeed')=['ooo','ee']
        test_len = [len(re.findall(r"(a{2,}|e{3,}|i{2,}|o{3,}|u{2,})", tweet.text_nopunc)) for tweet in tweets]  # length
        train_bool = [len(re.findall(r"(a{2,}|e{3,}|i{2,}|o{3,}|u{2,})", tweet.text_nopunc)) > 0 for tweet in
                      tweets_train]  # True or false
        test_bool = [len(re.findall(r"(a{2,}|e{3,}|i{2,}|o{3,}|u{2,})", tweet.text_nopunc)) > 0 for tweet in
                     tweets]  # true or false


        #https://stackoverflow.com/questions/20840803/how-to-convert-false-to-0-and-true-to-1-in-python
        train_bool=np.array([train_bool])*1#convert bool to binary 1 or 0
        test_bool=np.array([test_bool])*1#convert True to 1 and False to 0
        train=np.vstack((train_len,train_bool)).T
        test=np.vstack((test_len,test_bool)).T

        joblib.dump(train,trainfile,protocol=2)
        joblib.dump(test,testfile,protocol=2)
        return csr_matrix(train),csr_matrix(test),['len_vowelseq','bool_vowelseq']
    
    def find_capital_words(self,tweets,tweets_train):
        trainfile = self.pathfile+"train_" + "find_capital_words" + '.pkl'
        testfile = self.pathfile+"test_" + "find_capital_words" + '.pkl'

        if os.path.isfile(trainfile):
            train = joblib.load(trainfile)
            if os.path.isfile(testfile):
                test = joblib.load(testfile)
                return csr_matrix(train), csr_matrix(test), ['capital_words']

        train=[len(re.findall(r'[A-Z]+[a-z]+|[A-Z]+',tweet.text_nopunc)) for tweet in tweets_train]#
        test=[len(re.findall(r'[A-Z]+[a-z]+|[A-Z]+',tweet.text_nopunc)) for tweet in tweets]
        train=np.array([train]).T
        test=np.array([test]).T
        joblib.dump(train,trainfile,protocol=2)
        joblib.dump(test,testfile,protocol=2)

        return csr_matrix(train),csr_matrix(test),['capital_words']

    def num_punc(self,tweets,tweets_train):
        trainfile = self.pathfile+"train_" + "num_punc" + '.pkl'
        testfile = self.pathfile+"test_" + "num_punc" + '.pkl'

        if os.path.isfile(trainfile):
            train = joblib.load(trainfile)
            if os.path.isfile(testfile):
                test = joblib.load(testfile)
                return csr_matrix(train), csr_matrix(test), ["punc_?", "punc_!", "punc_,", "punc_.", "punc_;",
                                                             "punc_?!.,;"]

        train=[[len(re.findall(r'[?]',tweet.text_raw)),len(re.findall(r'[!]',tweet.text_raw)),len(re.findall(r'[.]',tweet.text_raw)),len(re.findall(r'[,]',tweet.text_raw)),\
            len(re.findall(r'[;]',tweet.text)),len(re.findall(r'[!?.,;]',tweet.text_raw))] for tweet in tweets_train]
        test=[[len(re.findall(r'[?]',tweet.text_raw)),len(re.findall(r'[!]',tweet.text_raw)),len(re.findall(r'[.]',tweet.text_raw)),len(re.findall(r'[,]',tweet.text_raw)),\
            len(re.findall(r'[;]',tweet.text_raw)),len(re.findall(r'[!?.,;]',tweet.text_raw))] for tweet in tweets]
        joblib.dump(train,trainfile,protocol=2)
        joblib.dump(test,testfile,protocol=2)
        return csr_matrix(train),csr_matrix(test),["punc_?","punc_!","punc_,","punc_.","punc_;","punc_?!.,;"]



    def get_word2vec(self,tweets,tweets_train,w2v_features,w2v_min_wc,w2v_ctxt,down_sampling,skip_cbow,Google_w2v,twitter_w2v):


        if Google_w2v==True:
            # https://groups.google.com/forum/#!topic/gensim/RQ7If14yWbk
            #model = KeyedVectors.load_word2vec_format(self.pathfile + 'GoogleNews-vectors-negative300.bin', binary=True)
            trainfile = self.pathfile + "train_" + "word2vec_" + 'gooole_nosim.pkl'
            testfile = self.pathfile + "test_" + "word2vec_" + 'google_nosim.pkl'
            featfile = self.pathfile + "feat_" + "word2vec_" +'google_nosim.pkl'

        elif twitter_w2v==True:

            # https://groups.google.com/forum/#!topic/gensim/RQ7If14yWbk
            #model = KeyedVectors.load_word2vec_format(self.pathfile + 'GoogleNews-vectors-negative300.bin', binary=True)
            trainfile = self.pathfile + "train_" + "word2vec_" + 'twitter_nosim.pkl'
            testfile = self.pathfile + "test_" + "word2vec_" + 'twitter_nosim.pkl'
            featfile = self.pathfile + "feat_" + "word2vec_" +'twitter_nosim.pkl'


        else:
            #changed
            trainfile = self.pathfile+"train_" + "get_word2vec" +'_'+str(skip_cbow)+'_'+str(w2v_features)+'_'+str(w2v_min_wc)+'_'+str(w2v_ctxt)+'_'+str(down_sampling)+ '.pkl'
            testfile = self.pathfile+"test_" + "get_word2vec" +'_'+str(skip_cbow)+'_'+str(w2v_features)+'_'+str(w2v_min_wc)+'_'+str(w2v_ctxt)+'_'+str(down_sampling)+ '.pkl'
            featfile = self.pathfile+"feat_" + "get_word2vec" +'_'+str(skip_cbow)+'_'+str(w2v_features)+'_'+str(w2v_min_wc)+'_'+str(w2v_ctxt)+'_'+str(down_sampling)+ '.pkl'


        if os.path.isfile(trainfile):
            tweets_train_matrix= joblib.load(trainfile)
            if os.path.isfile(testfile):
                tweets_matrix= joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat=joblib.load(featfile)

                    return tweets_train_matrix, tweets_matrix, feat

        word2vec_input=[]
        for tweet in tweets:
            word2vec_input.extend([tweet.word2vecinput])

        for tweet in tweets_train:
            word2vec_input.extend([tweet.word2vecinput])

        if Google_w2v==True:

            from gensim.models import KeyedVectors
            from time import time
            start=time()
            try:
                #print('1.......')
                model = word2vec.Word2Vec.load('vectors_google', mmap='r')
                #print('done')
            except:
                try:
                    #print('2.......')
                    model = KeyedVectors.load('vectors_google', mmap='r')
                    #print('done')
                except:
                    try:
                        #print('3.......')
                        model = KeyedVectors.load('vectors_google')
                        #print('done')
                    except:
                        try:
                            #print('4.......')
                            model = word2vec.Word2Vec.load("GoogleNews-vectors-negative300.bin")
                            #print('done')

                        except:
                            #print('5.......')
                            model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
                            #print('done')
                            #https://groups.google.com/forum/#!topic/gensim/RQ7If14yWbk
            print("loading Google model took %.2f seconds " % ((time() - start)))
        elif twitter_w2v==True:
            from gensim.models import KeyedVectors
            from time import time
            start=time()

            try:
                #print('1.......')
                model = word2vec.Word2Vec.load('vectors_twitter', mmap='r')
                #print('done')
            except:
                try:
                    #print('2.......')
                    model = KeyedVectors.load('vectors_twitter', mmap='r')
                    #print('done')

                except:
                    try:
                        #print('3.......')
                        model = KeyedVectors.load('vectors_twitter')
                        #print('done')

                    except:
                        try:
                            #print('4.......')
                            model = word2vec.Word2Vec.load("word2vec_twitter_model.bin")
                            #print('done')

                        except:
                            #print('5.......')
                            model = KeyedVectors.load_word2vec_format('word2vec_twitter_model.bin',binary=True, unicode_errors='ignore')#limit=1000000)
                            #print('done')

                            #https://groups.google.com/forum/#!topic/gensim/RQ7If14yWbk

            print("loading twitter model took %.2f seconds " % ((time() - start)))

        def tweet_to_vec(tokens,model,num_features,voc):
            tweet_vec=np.zeros((num_features),dtype="float32")
            nwords=0
            for word in tokens:
                try:
                    if word.lower() in voc:
                        tweet_vec+=model[word.lower()]
                        nwords+=1
                except:
                    #print("word not found in dictionary w2v : ",word)
                    pass
            if nwords==0:
                print('!!zero words ',tokens)
                tweet_vec=tweet_vec
            else:
                tweet_vec/=nwords
            return tweet_vec

        def generate_vectors(tweets,model,num_features,voc):
            curr_index = 0
            tweets_vecs = np.zeros((len(tweets), num_features), dtype="float32")
            for tweet in tweets:
                tweets_vecs[curr_index] = tweet_to_vec(tweet.tokens_nopunc, model, num_features,voc)

                curr_index += 1
            return tweets_vecs
        num_features=model.wv.syn0.shape[1]
        voc = set(model.wv.index2word)
        voc = set(e.lower() for e in voc)
        from time import time
        start=time()
        tweets_matrix=generate_vectors(tweets,model,num_features,voc)
        tweets_train_matrix=generate_vectors(tweets_train,model,num_features,voc)
        print("generating w2v features took %.2f seconds " % ((time() - start)))

        if Google_w2v==True:
            feat=['Google_w2v_'+str(i) for i in range(num_features)]
        elif twitter_w2v==True:
            feat=['twitter_w2v_'+str(i) for i in range(num_features)]
        else:
            feat=['w2v'+ str(skip_cbow) + '_' + str(
                    w2v_features) + '_' + str(w2v_min_wc) + '_' + str(w2v_ctxt) + '_' + str(down_sampling) +'_'+str(i) for i in range(num_features)]


            feat=['w2v'+ str(skip_cbow) + '_' + str(w2v_features) + '_' + str(w2v_min_wc) + '_' + str(w2v_ctxt) + '_' + str(down_sampling)]

        if Google_w2v==True:
            joblib.dump(tweets_train_matrix,self.pathfile + "train_" + "word2vec_" + 'gooole_nosim.pkl',protocol=2)
            joblib.dump(tweets_matrix,self.pathfile + "test_" + "word2vec_" + 'google_nosim.pkl',protocol=2)
            joblib.dump(feat,self.pathfile + "feat_" + "word2vec_" + 'google_nosim.pkl',protocol=2)
        elif twitter_w2v==True:
            joblib.dump(tweets_train_matrix,self.pathfile + "train_" + "word2vec_" + 'twitter_nosim.pkl',protocol=2)
            joblib.dump(tweets_matrix,self.pathfile + "test_" + "word2vec_" + 'twitter_nosim.pkl',protocol=2)
            joblib.dump(feat,self.pathfile + "feat_" + "word2vec_" + 'twitter_nosim.pkl',protocol=2)

        joblib.dump(tweets_train_matrix,trainfile,protocol=2)
        joblib.dump(tweets_matrix,testfile,protocol=2)
        joblib.dump(feat,featfile,protocol=2)
        return tweets_train_matrix,tweets_matrix,feat



    def get_ngrams(self,tweets,tweets_train,MINDF):
        trainfile = self.pathfile+"train_" + "get_ngrams" + '.pkl'
        testfile = self.pathfile+"test_" + "get_ngrams" + '.pkl'
        featfile=self.pathfile+"feat_"+ "get_ngrams" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
            if os.path.isfile(featfile):
                feat=joblib.load(featfile)
                return trainmatrix,testmatrix,feat

        vec=CountVectorizer(ngram_range=(1, 3),min_df=MINDF)



        feat_test=[]
        feat_train=[]
        for tweet in tweets_train:
            feat_train.append(tweet.text_nopunc)
        for tweet in tweets:
            feat_test.append(tweet.text_nopunc)

        vec.fit_transform(feat_train)
        trainmatrix=vec.transform(feat_train)
        testmatrix=vec.transform(feat_test)
        feat=vec.get_feature_names()
        #print("number of features ",len(feat))

        feat = ["ngrams_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix,trainfile,protocol=2)
        joblib.dump(testmatrix,testfile,protocol=2)
        joblib.dump(feat,featfile,protocol=2)
        return trainmatrix,testmatrix,feat #####

    def get_ngrams_idf(self,tweets,tweets_train,MINDF):#uwb#
        trainfile = self.pathfile+"train_" + "get_ngrams_idf" + '.pkl'
        testfile = self.pathfile+"test_" + "get_ngrams_idf" + '.pkl'
        featfile=self.pathfile+"feat_"+ "get_ngrams_idf" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat=joblib.load(featfile)
                    return trainmatrix, testmatrix, feat  #####

        vec=TfidfVectorizer(ngram_range=(1, 3),min_df=MINDF)#

        feat_test=[]
        feat_train=[]
        for tweet in tweets_train:
            feat_train.append(tweet.text_nopunc)
        for tweet in tweets:
            feat_test.append(tweet.text_nopunc)

        vec.fit_transform(feat_train)
        trainmatrix=vec.transform(feat_train)
        testmatrix=vec.transform(feat_test)
        feat=vec.get_feature_names()
        #print("number of features ", len(feat))

        feat = ["ngramsidf_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix,trainfile,protocol=2)
        joblib.dump(testmatrix,testfile,protocol=2)
        joblib.dump(feat,featfile,protocol=2)
        return trainmatrix,testmatrix,feat #####
    
    def get_char(self,tweets,tweets_train,n,MINDF):
        import scipy

        trainfile = self.pathfile+"train_" + "get_char" + '.pkl'
        testfile = self.pathfile+"test_" + "get_char" + '.pkl'
        featfile= self.pathfile+"feat_"+ "get_char" + '.pkl'


        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat=joblib.load(featfile)
                    return csr_matrix(trainmatrix),csr_matrix(testmatrix), feat

        feat_test=[]
        feat_train=[]

        for tweet in tweets_train:
            feat_train.append(" ".join(tweet.text_nopunc.split()))#
        for tweet in tweets:
            feat_test.append(" ".join(tweet.text_nopunc.split()))
        countvec=CountVectorizer(analyzer="char",ngram_range=(1,6),min_df=MINDF)

        countvec.fit_transform(feat_train)
        trainmatrix=countvec.transform(feat_train)
        testmatrix=countvec.transform(feat_test)
        feat=countvec.get_feature_names()
        #print("number of features ",len(feat))
        try:
            ind_space=feat.index(u' ')
            feat.remove(u' ')
            trainmatrix = np.delete(trainmatrix.toarray(),[ind_space],axis=1)
            testmatrix = np.delete(testmatrix.toarray(),[ind_space],axis=1)
        except: pass

        feat = ["nchars_" + re.sub(" ", "_", e) for e in feat]

        joblib.dump(trainmatrix,trainfile,protocol=2)
        joblib.dump(testmatrix,testfile,protocol=2)
        joblib.dump(feat,featfile,protocol=2)

        return trainmatrix,testmatrix,feat

    
    def get_hashtag(self,tweets,tweets_train,MINDF):

        trainfile = self.pathfile+"train_" + "get_hashtag" + '.pkl'
        testfile = self.pathfile+"test_" + "get_hashtag" + '.pkl'
        featfile = self.pathfile+"feat_" + "get_hashtag" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat


        feat_train = []
        for t in tweets_train:
            hash = ""
            hashtag = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
            for elem in hashtag:
                elem = elem.lstrip('<hashtag>');
                elem = elem.rstrip('</hashtag>');
                elem = elem.strip()
                hash += elem + " "
            feat_train.append(hash)
        feat_test= []
        for t in tweets:
            hash = ""
            hashtag = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
            for elem in hashtag:
                elem = elem.lstrip('<hashtag>');
                elem = elem.rstrip('</hashtag>');
                elem = elem.strip()
                hash += elem + " "
            feat_test.append(hash)

        countvec=CountVectorizer(ngram_range=(1,3),min_df=MINDF)#
        countvec.fit_transform(feat_train)
        trainmatrix=countvec.transform(feat_train)
        testmatrix=countvec.transform(feat_test)
        feat=countvec.get_feature_names()
        #print("number of features ",len(feat))

        feat = ["hashtag_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix,trainfile,protocol=2)
        joblib.dump(testmatrix,testfile,protocol=2)
        joblib.dump(feat,featfile,protocol=2)

        return trainmatrix,testmatrix,feat

    def get_hashtag_idf(self,tweets,tweets_train,MINDF):
        #uwb paper
        trainfile = self.pathfile+"train_" + "get_hashtag_idf" + '.pkl'
        testfile = self.pathfile+"test_" + "get_hashtag_idf" + '.pkl'
        featfile = self.pathfile+"feat_" + "get_hashtag_idf" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat


        feat_train = []
        for t in tweets_train:
            hash = ""
            hashtag = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
            for elem in hashtag:
                elem = elem.lstrip('<hashtag>');
                elem = elem.rstrip('</hashtag>');
                elem = elem.strip()
                hash += elem + " "
            feat_train.append(hash)
        feat_test= []
        for t in tweets:
            hash = ""
            hashtag = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
            for elem in hashtag:
                elem = elem.lstrip('<hashtag>');
                elem = elem.rstrip('</hashtag>');
                elem = elem.strip()
                hash += elem + " "
            feat_test.append(hash)


        vec=TfidfVectorizer(ngram_range=(1,3),min_df=MINDF)

        #http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        vec.fit_transform(feat_train)
        trainmatrix=vec.transform(feat_train)
        testmatrix=vec.transform(feat_test)
        feat=vec.get_feature_names()
        #print("number of features ", len(feat))

        feat = ["hashtagidf_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix,trainfile,protocol=2)
        joblib.dump(testmatrix,testfile,protocol=2)
        joblib.dump(feat,featfile,protocol=2)

        return trainmatrix,testmatrix,feat

    def get_screen(self,tweets,tweets_train,MINDF):
        trainfile = self.pathfile+"train_" + "get_screen" + '.pkl'
        testfile = self.pathfile+"test_" + "get_screen" + '.pkl'
        featfile = self.pathfile+"feat_" + "get_screen" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat



        feat_train = []
        for t in tweets_train:
            user = ""
            users = re.findall(r'<user>[\s+\w+\._?!]+</user>', t.text_hash)
            for elem in users:
                elem = elem.lstrip('<user>');
                elem = elem.rstrip('</user>');
                elem = elem.strip()
                user += elem + " "
            feat_train.append(user)
        feat_test=[]
        for t in tweets:
            user = ""
            users = re.findall(r'<user>[\s+\w+\._?!]+</user>', t.text_hash)
            for elem in users:
                elem = elem.lstrip('<user>');
                elem = elem.rstrip('</user>');
                elem = elem.strip()
                user += elem + " "
            feat_test.append(user)

        countvec=CountVectorizer(ngram_range=(1,2),min_df=MINDF)
        countvec.fit_transform(feat_train)
        trainmatrix=countvec.transform(feat_train)
        testmatrix=countvec.transform(feat_test)
        feat=countvec.get_feature_names()
        #print("number of features ",len(feat))

        feat = ["screen_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix,trainfile,protocol=2)
        joblib.dump(testmatrix,testfile,protocol=2)
        joblib.dump(feat,featfile,protocol=2)

        return trainmatrix,testmatrix,feat
        

    def get_num_hashtag(self,tweets,tweets_train):
        trainfile = self.pathfile+"train_" + "get_num_hashtag" + '.pkl'
        testfile = self.pathfile+"test_" + "get_num_hashtag" + '.pkl'

        if os.path.isfile(trainfile):
            feat_train =joblib.load(trainfile)
            if os.path.isfile(testfile):
                feat_test=joblib.load(testfile)
                return csr_matrix(np.vstack(feat_train)), csr_matrix(np.vstack(feat_test)), ["num_hashtag"]

        feat_test=[]
        feat_train=[]
        for tweet in tweets_train:
            r=re.findall(r'#',tweet.text_raw)#
            feat_train.append(len(r))
        for tweet in tweets:
            r=re.findall(r'#',tweet.text_raw)#

            feat_test.append(len(r))
        joblib.dump(feat_train,trainfile,protocol=2)
        joblib.dump(feat_test,testfile,protocol=2)

        return csr_matrix(np.vstack(feat_train)),csr_matrix(np.vstack(feat_test)),["num_hashtag"]

    def get_num_screen(self,tweets,tweets_train):

        trainfile = self.pathfile+"train_" + "get_num_screen" + '.pkl'
        testfile = self.pathfile+"test_" + "get_num_screen" + '.pkl'

        if os.path.isfile(trainfile):
            feat_train = joblib.load(trainfile)
            if os.path.isfile(testfile):
                feat_test = joblib.load(testfile)
                return csr_matrix(np.vstack(feat_train)), csr_matrix(np.vstack(feat_test)), ["num_screen"]

        feat_test=[]
        feat_train=[]
        for tweet in tweets_train:
            r=re.findall(r'@',tweet.text_raw)#
            feat_train.append(len(r))
        for tweet in tweets:
            r=re.findall(r'@',tweet.text_raw)#
            feat_test.append(len(r))
        joblib.dump(feat_train,trainfile,protocol=2)
        joblib.dump(feat_test,testfile,protocol=2)
        return csr_matrix(np.vstack(feat_train)),csr_matrix(np.vstack(feat_test)),["num_screen"]


    def bow_hashtag(self,tweets,tweets_train,against_list,favor_list,target_list,MINDF):

        trainfile = self.pathfile+"train_" + "bow_hashtag" + '.pkl'
        testfile = self.pathfile+"test_" + "bow_hashtag" + '.pkl'
        featfile = self.pathfile+"feat_" + "bow_hashtag" + '.pkl'
        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(testfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat


        hashtags_train = []
        for t in tweets_train:
            hash = ""
            hashtag = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
            for elem in hashtag:
                elem = elem.lstrip('<hashtag>');
                elem = elem.rstrip('</hashtag>');
                elem = elem.strip()
                hash += elem + " "
            hashtags_train.append(hash)
        hashtags_test = []
        for t in tweets:
            hash = ""
            hashtag = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
            for elem in hashtag:
                elem = elem.lstrip('<hashtag>');
                elem = elem.rstrip('</hashtag>');
                elem = elem.strip()
                hash += elem + " "
            hashtags_test.append(hash)
        Words = ""
        for e in target_list:
            Words += e + "|"
        Words = Words[:-1]
        words_train = words_test = Words

        hash_train=[re.sub(words_train,"TARGET",tag,flags=re.IGNORECASE) for tag in hashtags_train]

        hash_test=[re.sub(words_test,"TARGET",tag,flags=re.IGNORECASE) for tag in hashtags_test]

        countvec=CountVectorizer(ngram_range=(1,3),min_df=MINDF)#

        countvec.fit_transform(hash_train)
        trainmatrix=countvec.transform(hash_train)
        testmatrix=countvec.transform(hash_test)
        feat=countvec.get_feature_names()
        #print("number of features ", len(feat))

        feat = ["bowhashtag_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix,trainfile,protocol=2)
        joblib.dump(testmatrix,testfile,protocol=2)
        joblib.dump(feat,featfile,protocol=2)

        return trainmatrix,testmatrix,feat

    def bow_mention(self,tweets,tweets_train,against_list,favor_list,target_list,MINDF):

        trainfile =self.pathfile+ "train_" + "bow_mention" + '.pkl'
        testfile = self.pathfile+"test_" + "bow_mention" + '.pkl'
        featfile = self.pathfile+"feat_" + "bow_mention" + '.pkl'
        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(testfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat


        mention_train = []
        for t in tweets_train:
            hash = ""
            hashtag = re.findall(r'<user>[\s+\w+\._?!]+</user>', t.text_hash)
            for elem in hashtag:
                elem = elem.lstrip('<user>');
                elem = elem.rstrip('</user>');
                elem = elem.strip()
                hash += elem + " "
            mention_train.append(hash)
        mention_test = []
        for t in tweets:
            hash = ""
            hashtag = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
            for elem in hashtag:
                elem = elem.lstrip('<hashtag>');
                elem = elem.rstrip('</hashtag>');
                elem = elem.strip()
                hash += elem + " "
            mention_test.append(hash)

        words="hillary|clinton|hc"

        inoppwords = "Bernie|Sanders|Martin|O'Malley|Lincoln|Chafee|Webb|Lawrence|Lessig"
        outoppwords = "republican|republicans|conservative|realDonaldTrump|Donald|Trump|ted|cruz|Marco|Rubio|John|Kasich\
                Ben|Carson|Jeb|Bush|Rand|Paul|Mike|Huckabee|Carly|Fiorina|Chris|Christie|Rick|Santorum|Gilmore|Rick|Perry|Scott|Walker\
                Bobby|Jindal|Lindsey|Graham|George|Pataki"
        meopp = "benghazi|fell the bern|bern"

        Words = ""
        for e in target_list:
            Words += e + "|"
        Words = Words[:-1]
        words_train = words_test = Words




        ment_train=[re.sub(words_train,"TARGET",tag,flags=re.IGNORECASE) for tag in mention_train]


        ment_test=[re.sub(words_test,"TARGET",tag,flags=re.IGNORECASE) for tag in mention_test]
        countvec = CountVectorizer(ngram_range=(1, 2),min_df=MINDF)
        countvec.fit_transform(ment_train)
        trainmatrix=countvec.transform(ment_train)
        testmatrix=countvec.transform(ment_test)
        feat=countvec.get_feature_names()
        #print("number of features ", len(feat))

        feat = ["bowmention_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix,trainfile,protocol=2)
        joblib.dump(testmatrix,testfile,protocol=2)
        joblib.dump(feat,featfile,protocol=2)

        return trainmatrix,testmatrix,feat
    
    def pos_feat(self,tweets,tweets_train,MINDF):

        trainfile = self.pathfile+"train_" + "pos_feat" + '.pkl'
        testfile = self.pathfile+"test_" + "pos_feat" + '.pkl'
        featfile = self.pathfile+"feat_" + "pos_feat" + '.pkl'
        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(testfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat

        features=[]
        for tweet in tweets:
            feature=" ".join([token[0]+"_"+token[1] for token in nltk.pos_tag(tweet.tokens_nopunc)])
            features.append(feature)
        features_train=[]
        for tweet in tweets_train:
            feature=" ".join([token[0]+"_"+token[1] for token in nltk.pos_tag(tweet.tokens_nopunc)])
            features_train.append(feature)
            

        countvec = CountVectorizer(ngram_range=(1, 3),min_df=MINDF)
        countvec.fit_transform(features_train)
        trainmatrix=countvec.transform(features_train)
        testmatrix=countvec.transform(features)
        feat=countvec.get_feature_names()
        #print("number of features ", len(feat))

        feat = ["posfeat_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix,trainfile,protocol=2)
        joblib.dump(testmatrix,testfile,protocol=2)
        joblib.dump(feat,featfile,protocol=2)

        return trainmatrix,testmatrix,feat
    def pos_feat_1(self,tweets,tweets_train,MINDF):

        trainfile = self.pathfile+"train_" + "pos_feat_1" + '.pkl'
        testfile = self.pathfile+"test_" + "pos_feat_1" + '.pkl'
        featfile = self.pathfile+"feat_" + "pos_feat_1" + '.pkl'
        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(testfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat
        features=[]
        for tweet in tweets:

            feature0=" ".join([token[0] for token in nltk.pos_tag(tweet.tokens_nopunc)])
            feature1=" ".join([token[1] for token in nltk.pos_tag(tweet.tokens_nopunc)])
            feature=feature0+" "+feature1
            features.append(feature)
        features_train=[]
        for tweet in tweets_train:

            feature0=" ".join([token[0] for token in nltk.pos_tag(tweet.tokens_nopunc)])
            feature1=" ".join([token[1] for token in nltk.pos_tag(tweet.tokens_nopunc)])
            feature=feature0+" "+feature1
            features_train.append(feature)
            
        #print features
        #countvec=CountVectorizer(ngram_range=(1,2),max_features=750)
        countvec = CountVectorizer(ngram_range=(1, 3),min_df=MINDF)
        countvec.fit_transform(features_train)
        trainmatrix=countvec.transform(features_train)
        testmatrix=countvec.transform(features)
        feat=countvec.get_feature_names()
        #print("number of features ", len(feat))

        feat = ["posfeat1_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix,trainfile,protocol=2)
        joblib.dump(testmatrix,testfile,protocol=2)
        joblib.dump(feat,featfile,protocol=2)

        return trainmatrix,testmatrix,feat
    def pos_feat_2(self,tweets,tweets_train,MINDF):#uwb

        trainfile = self.pathfile+"train_" + "pos_feat_2" + '.pkl'
        testfile = self.pathfile+"test_" + "pos_feat_2" + '.pkl'
        featfile = self.pathfile+"feat_" + "pos_feat_2" + '.pkl'
        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix =joblib.load(testfile)
                if os.path.isfile(testfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat
        test=[];train=[]
        for tweet in tweets:
            feature=" ".join([token[1] for token in nltk.pos_tag(tweet.tokens_nopunc)])
            test.append(feature)
        features_train=[]
        for tweet in tweets_train:
            feature=" ".join([token[1] for token in nltk.pos_tag(tweet.tokens_nopunc)])
            train.append(feature)
            
        vec=TfidfVectorizer(ngram_range=(1,3),min_df=MINDF)
        trainmatrix=vec.fit_transform(train)
        testmatrix=vec.transform(test)
        feat=vec.get_feature_names()
        #print("number of features ", len(feat))

        feat = ["posfeat2_" + re.sub(" ", "_", e) for e in feat]

        joblib.dump(trainmatrix,trainfile,protocol=2)
        joblib.dump(testmatrix,testfile,protocol=2)
        joblib.dump(feat,featfile,protocol=2)

        return trainmatrix,testmatrix,feat



    def get_dal(self,tweets,tweets_train):
        feat=["dal_pleasantness_avg","dal_activation_avg","dal_imagery_avg","dal_pleasantness_sum","dal_activation_sum","dal_imagery_sum"]

        trainfile =self.pathfile+ "train_" + "get_dal" + '.pkl'
        testfile = self.pathfile+"test_" + "get_dal" + '.pkl'
        if os.path.isfile(trainfile):
            dal_train = joblib.load(trainfile)
            if os.path.isfile(testfile):
                dal_test = joblib.load(testfile)
                return csr_matrix(dal_train), csr_matrix(dal_test), feat
        dal_train=[]
        dal_test=[]
        
        for tweet in tweets_train:
            dal_train.append(tweet.sentimentdal)
        for tweet in tweets:
            dal_test.append(tweet.sentimentdal)

        joblib.dump(dal_train,trainfile,protocol=2)
        joblib.dump(dal_test,testfile,protocol=2)



        return csr_matrix(np.vstack(dal_train)),csr_matrix(np.vstack(dal_test)),feat
    
    def get_gi(self,tweets,tweets_train):

        feat=["gi_pos","gi_neg","gi_host","gi_strong","gi_pleas","gi_pain"]

        trainfile = self.pathfile+"train_" + "get_gi" + '.pkl'
        testfile = self.pathfile+"test_" + "get_gi" + '.pkl'
        if os.path.isfile(trainfile):
            gi_train = joblib.load(trainfile)
            if os.path.isfile(testfile):
                gi_test = joblib.load(testfile)
                return csr_matrix(gi_train), csr_matrix(gi_test), feat

        gi_train=[]
        gi_test=[]
        
        for tweet in tweets_train:
            gi_train.append(tweet.sentimentgi)
        for tweet in tweets:
            gi_test.append(tweet.sentimentgi)
        joblib.dump(gi_train,trainfile,protocol=2)
        joblib.dump(gi_test,testfile,protocol=2)


        return csr_matrix(np.vstack(gi_train)),csr_matrix(np.vstack(gi_test)),feat

    def get_afinn(self,tweets,tweets_train,scaling="yes"):
        trainfile = self.pathfile+"train_" + "get_afinn" + '.pkl'
        testfile = self.pathfile+"test_" + "get_afinn" + '.pkl'
        if os.path.isfile(trainfile):
            afinn_train = joblib.load(trainfile)
            if os.path.isfile(testfile):
                afinn_test = joblib.load(testfile)
                return csr_matrix(afinn_train).T, csr_matrix(afinn_test).T, ["sentiment_afinn"]
        afinn_train=[]
        afinn_test=[]

        if scaling=="yes":
            for tweet in tweets_train:
                afinn_train.append(float(tweet.sentimentafinn)/5)
            for tweet in tweets:
                afinn_test.append(float(tweet.sentimentafinn)/5)
        elif scaling=="no":
            for tweet in tweets_train:
                afinn_train.append(tweet.sentimentafinn)
            for tweet in tweets:
                afinn_test.append(tweet.sentimentafinn)
        joblib.dump(afinn_train,trainfile,protocol=2)
        joblib.dump(afinn_test,testfile,protocol=2)

        return csr_matrix(np.vstack(afinn_train)),csr_matrix(np.vstack(afinn_test)),["sentiment_afinn"]

    def mpqa(self,tweets,tweets_train):#takelab_paper
        trainfile = self.pathfile+"train_" + "mpqa" + '.pkl'
        testfile = self.pathfile+"test_" + "mpqa" + '.pkl'
        if os.path.isfile(trainfile):
            train_mat = joblib.load(trainfile)
            if os.path.isfile(testfile):
                test_mat = joblib.load(testfile)
                return csr_matrix(train_mat), csr_matrix(test_mat), ["mpqa_sum", "mpqa_pr", "mpqa_nr"]

        train=[tweet.tokens_nopunc for tweet in tweets_train]
        test=[tweet.tokens_nopunc for tweet in tweets]
        #voca = codecs.open('lexicons/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff', 'r').read().splitlines()
        voca = codecs.open('resources/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff', 'r').read().splitlines()
        wds = {}
        for i in voca:
            i = i.split()
            try:#
                if wds1[i[2].split('=')[1]] != i[5].split('=')[1]:#
                    pass
            except:
                if i[5].split('=')[1] in ['positive', 'negative']:
                    wds[i[2].split('=')[1]] = i[5].split('=')[1]

        pr_train=[];nr_train=[];v_train=[]       
        for tweet in train:
            pos=0;neg=0
            for token in tweet:
                if token.lower() in wds:
                    if wds[token.lower()]=='positive':
                        pos+=1
                    elif wds[token.lower()]=='negative':
                        neg+=1
            
            v_train.append(pos-neg)#todo
            pr_train.append(float(pos)/len(tweet))
            nr_train.append(float(neg)/len(tweet))
            
    
        pr_test=[];nr_test=[];v_test=[];

        for tweet in test:
            pos=0;neg=0
            for token in tweet:
                if token.lower() in wds:
                    if wds[token.lower()]=='positive':
                        pos+=1
                    elif wds[token.lower()]=='negative':
                        neg+=1
            v_test.append(pos-neg)
            pr_test.append(float(pos)/len(tweet))
            nr_test.append(float(neg)/len(tweet))
            
        train_mat=np.vstack((v_train, pr_train,nr_train)).T
        test_mat=np.vstack((v_test, pr_test,nr_test)).T
        joblib.dump(train_mat,trainfile,protocol=2)
        joblib.dump(test_mat,testfile,protocol=2)
        return csr_matrix(train_mat),csr_matrix(test_mat),["mpqa_sum","mpqa_pr","mpqa_nr"]

    def bing_lius(self,tweets,tweets_train):#takelab_paper
        trainfile = self.pathfile+"train_" + "bing_lius" + '.pkl'
        testfile = self.pathfile+"test_" + "bing_lius" + '.pkl'
        if os.path.isfile(trainfile):
            train_mat = joblib.load(trainfile)
            if os.path.isfile(testfile):
                test_mat = joblib.load(testfile)
                return csr_matrix(train_mat), csr_matrix(test_mat) ,["binglius_sum","binglius_pr","binglius_nr"]

        import codecs
        #with codecs.open('lexicons/positive-words_bing_liu.txt', 'r') as inFile:
        with codecs.open('resources/positive-words_bing_liu.txt', 'r') as inFile:
            positive = set(inFile.read().splitlines())
        #with codecs.open('lexicons/negative-words_bing_liu.txt', 'r') as inFile:
        with codecs.open('resources/negative-words_bing_liu.txt', 'r') as inFile:
            negative = set(inFile.read().splitlines())

        train=[tweet.tokens_nopunc for tweet in tweets_train]
        test=[tweet.tokens_nopunc for tweet in tweets]

        pr_train=[];nr_train=[];v_train=[]

        for tweet in train:
            pos=0;neg=0
            for token in tweet:
                if token.lower() in positive:
                    pos+=1
                elif token.lower() in negative:
                    neg+=1

            v_train.append(pos-neg)#todo
            pr_train.append(float(pos)/len(tweet))
            nr_train.append(float(neg)/len(tweet))

        pr_test=[];nr_test=[];v_test=[];

        for tweet in test:
            pos=0;neg=0;
            for token in tweet:
                if token.lower() in positive:
                    pos+=1
                elif token.lower() in negative:
                    neg+=1
            v_test.append(pos-neg)
            pr_test.append(float(pos)/len(tweet))
            nr_test.append(float(neg)/len(tweet))

        train_mat=np.vstack((v_train, pr_train,nr_train)).T
        test_mat=np.vstack((v_test, pr_test,nr_test)).T
        joblib.dump(train_mat,trainfile,protocol=2)
        joblib.dump(test_mat,testfile,protocol=2)
        return csr_matrix(train_mat),csr_matrix(test_mat),["binglius_sum","binglius_pr","binglius_nr"]

    def NRC_hashtag(self,tweets,tweets_train):
        feat=["NRC_tag_sum","NRC_tag_max","NRC_tag_min","NRC_tag_pr","NRC_tag_nr"]
        trainfile = self.pathfile+"train_" + "NRC_hashtag" + '.pkl'
        testfile = self.pathfile+"test_" + "NRC_hashtag" + '.pkl'
        if os.path.isfile(trainfile):
            train_mat = joblib.load(trainfile)
            if os.path.isfile(testfile):
                test_mat = joblib.load(testfile)
                return csr_matrix(train_mat), csr_matrix(test_mat),feat

        import codecs
        #path2lexicon='lexicons/NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt'
        path2lexicon='resources/NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt'
        with codecs.open(path2lexicon, 'r') as inFile:
            wds = inFile.read().splitlines()#
        lex={}
        for i in wds:
            line=i.split("\t")
            lex[line[0]]=float(line[1])
        train=[tweet.text_raw.split() for tweet in tweets_train]
        test=[tweet.text_raw.split() for tweet in tweets]

        allv_train=[]#all pos neg values for all tweets , a list of list
        v_train=[]#a list , for each tweet
        maxv_train=[]#max score in each tweet#a list
        minv_train=[]#min score
        sumv_train=[]#sum of scores for each tweet
        pr_train=[]#ratio of positive words to all words in tweet
        nr_train=[]#ratio of negative words to all words in tweet        

        for tweet in train:
            v_train=[0];pos=0;neg=0
            for word in tweet:
                if word.lower() in lex:
                    v_train.append(lex[word.lower()])
                    if float(lex[word.lower()])>0:
                        pos+=1
                    elif float(lex[word.lower()])<0:
                        neg+=1
            sumv_train.append(sum(v_train))
            maxv_train.append(max(v_train))
            minv_train.append(min(v_train))
            pr_train.append(float(pos)/len(tweet))
            nr_train.append(float(neg)/len(tweet))
            allv_train.append(v_train)

        allv_test=[]#all pos neg values for all tweets , a list of list
        v_test=[]#a list , for each tweet
        maxv_test=[]#max score in each tweet#a list
        minv_test=[]#min score
        sumv_test=[]#sum of scores for each tweet
        pr_test=[]#ratio of positive words to all words in tweet
        nr_test=[]#ratio of negative words to all words in tweet        

        for tweet in test:
            v_test=[0];pos=0;neg=0
            for word in tweet:
                if word.lower() in lex:
                    v_test.append(lex[word.lower()])
                    if float(lex[word.lower()])>0:
                        pos+=1
                    elif float(lex[word.lower()])<0:
                        neg+=1
            sumv_test.append(sum(v_test))
            maxv_test.append(max(v_test))
            minv_test.append(min(v_test))
            pr_test.append(float(pos)/len(tweet))
            nr_test.append(float(neg)/len(tweet))
            allv_test.append(v_test)

        train_mat=np.vstack((sumv_train,maxv_train,minv_train,pr_train,nr_train)).T
        test_mat=np.vstack((sumv_test,maxv_test,minv_test,pr_test,nr_test)).T
        joblib.dump(train_mat,trainfile,protocol=2)
        joblib.dump(test_mat,testfile,protocol=2)
            
        return csr_matrix(train_mat),csr_matrix(test_mat),feat

    def NRC_sent140(self,tweets,tweets_train):#takelab_paper
        feat=["sent140_sum","sent140_max","sent140_min","sent140_pr","sent140_nr"]
        trainfile = self.pathfile+"train_" + "NRC_sent140" + '.pkl'
        testfile = self.pathfile+"test_" + "NRC_sent140" + '.pkl'
        if os.path.isfile(trainfile):
            train_mat =joblib.load(trainfile)
            if os.path.isfile(testfile):
                test_mat = joblib.load(testfile)
                return csr_matrix(train_mat), csr_matrix(test_mat),feat

        import codecs
        #path2lexicon='D:/Mycode/lexicons/Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt'
        path2lexicon='resources/Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt'
        with codecs.open(path2lexicon, 'r') as inFile:
            wds = inFile.read().splitlines()


        lex={}
        for i in wds:
            line=i.split("\t")
            lex[line[0]]=float(line[1])


        train = [tweet.text_raw.split() for tweet in tweets_train]
        test = [tweet.text_raw.split() for tweet in tweets]
            
        allv_train=[]#all pos neg values for all tweets , a list of list
        v_train=[]#a list , for each tweet
        maxv_train=[]#max score in each tweet#a list
        minv_train=[]#min score
        sumv_train=[]#sum of scores for each tweet
        pr_train=[]#ratio of positive words to all words in tweet
        nr_train=[]#ratio of negative words to all words in tweet        

        for tweet in train:
            v_train=[0];pos=0;neg=0
            for word in tweet:
                if word.lower() in lex:
                    v_train.append(lex[word.lower()])
                    if float(lex[word.lower()])>0:
                        pos+=1
                    elif float(lex[word.lower()])<0:
                        neg+=1
            sumv_train.append(sum(v_train))
            maxv_train.append(max(v_train))
            minv_train.append(min(v_train))
            pr_train.append(float(pos)/len(tweet))
            nr_train.append(float(neg)/len(tweet))
            allv_train.append(v_train)

        allv_test=[]#all pos neg values for all tweets , a list of list
        v_test=[]#a list , for each tweet
        maxv_test=[]#max score in each tweet#a list
        minv_test=[]#min score
        sumv_test=[]#sum of scores for each tweet
        pr_test=[]#ratio of positive words to all words in tweet
        nr_test=[]#ratio of negative words to all words in tweet        

        for tweet in test:
            v_test=[0];pos=0;neg=0
            for word in tweet:
                if word.lower() in lex:
                    v_test.append(lex[word.lower()])
                    if float(lex[word.lower()])>0:
                        pos+=1
                    elif float(lex[word.lower()])<0:
                        neg+=1
            sumv_test.append(sum(v_test))
            maxv_test.append(max(v_test))
            minv_test.append(min(v_test))
            pr_test.append(float(pos)/len(tweet))
            nr_test.append(float(neg)/len(tweet))
            allv_test.append(v_test)
            
        train_mat=np.vstack((sumv_train,maxv_train,minv_train,pr_train,nr_train)).T
        test_mat=np.vstack((sumv_test,maxv_test,minv_test,pr_test,nr_test)).T
        joblib.dump(train_mat,trainfile,protocol=2)
        joblib.dump(test_mat,testfile,protocol=2)
            
        return csr_matrix(train_mat),csr_matrix(test_mat),feat


    def find_neg(self,tweets,tweets_train):#takelab paper
        trainfile = self.pathfile+"train_" + "find_neg" + '.pkl'
        testfile = self.pathfile+"test_" + "find_neg" + '.pkl'
        if os.path.isfile(trainfile):
            train = joblib.load(trainfile)
            if os.path.isfile(testfile):
                test = joblib.load(testfile)
                return csr_matrix(train), csr_matrix(test), ["neg"]

        neg=['no','cannot','not','none','nothing','nowhere','neither','nor','nobody','hardly',"scarcely","barely","never","n\'t","none","havent","hasnt","cant",\
           "hadnt","couldnt","shouldnt","wont","wouldnt","dont","doesnt","didnt","arent","isnt","aint"]
        pattern=""; i=0
        for e in neg:
            if i==0:
                pattern=e
                i=1
            else:
                pattern=pattern+"|"+e

        train = [len(re.findall(pattern, tweet.text_nopunc, flags=re.IGNORECASE)) for tweet in tweets_train]
        test = [len(re.findall(pattern, tweet.text_nopunc, flags=re.IGNORECASE)) for tweet in tweets]
        train=np.array([train]).T
        test=np.array([test]).T
        joblib.dump(train,trainfile,protocol=2)
        joblib.dump(test,testfile,protocol=2)
        return csr_matrix(train),csr_matrix(test),["neg"]



    def stan_parser(self,tweets,tweets_train,MINDF):

        trainfile = self.pathfile+"train_" + "stan_parser" + '.pkl'
        testfile = self.pathfile+"test_" + "stan_parser" + '.pkl'
        featfile = self.pathfile+"feat_" + "stan_parser" + '.pkl'
        if os.path.isfile(trainfile):
            train_matrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                test_matrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat_train = joblib.load(featfile)
                    return csr_matrix(train_matrix), csr_matrix(test_matrix), feat_train

        import stanford_parser as sp



        train = [tweet.text_nopunc for tweet in
                 tweets_train]
        test = [tweet.text_nopunc for tweet in tweets]


        parser=sp.stanfordParser()


        def get_dependency(tweets,train_test):


            all_dep_list=[]
            for tweet in tweets:
                sent=parser.tokenize(tweet)
                ind = sp.get_offset(tweet, [ s["words"] for s in sent] );

                for i in range(len(sent)):
                    sent[i]["index"] = ind[i];

                for i in range(len(sent)):
                    data = parser.parse(sent[0]);
                    all_dep_list.append([data['org_dependency']])



            tweets_dep=[]
            all_dep=set()
            for e in all_dep_list:
                for i in e:

                    tweet_depcy=[]
                    dep_list=i.toString().split('),')#convert Java array list to string
                    for c in dep_list:

                        this_dep=[x.lower() for x in re.findall(r'[A-Za-z_]+', c)]#a tuple to add to a set
                        this_dep=tuple(this_dep)
                        tweet_depcy.append(this_dep)
                        all_dep.add(this_dep)

                tweets_dep.append(tweet_depcy)#list of list



            if train_test=="train":
                all_dep_noduplicate=list(all_dep)

                return tweets_dep,all_dep_noduplicate

            elif train_test=="test":
                return tweets_dep

        def rel_gov_dep(tweets_dep,all_dep_noduplicate,length,train_test,rel_gov_all=None,gov_dep_all=None):


            relgovdepmat=np.zeros([length,len( all_dep_noduplicate)])
            i=0
            for dep in  tweets_dep:#each tweet list

                for e in dep:
                    if e in all_dep_noduplicate:
                        ind=all_dep_noduplicate.index(e)
                        relgovdepmat[i][ind]=1

                i+=1

            rel_gov=[]
            #print("tweets_dep ",tweets_dep)
            for e in tweets_dep:
                t=[]
                for dep in e:
                    t.append(dep[:-1])
                rel_gov.append(t)
            if train_test=="train":
                rel_gov_all = set()
                for e in all_dep_noduplicate:
                    rel_gov_all.add(e[:-1])
                rel_gov_all=list(rel_gov_all)

            relgovmat = np.zeros([length, len(rel_gov_all)])
            i=0
            for dep in rel_gov:#each tweet list
                for e in dep:
                    if e in rel_gov_all:
                        ind=rel_gov_all.index(e)
                        relgovmat[i][ind]=1
                    #print("relgovmat ",relgovmat)

                i+=1

            gov_dep=[]
            for e in tweets_dep:
                t=[]
                for dep in e:
                    t.append(dep[1:])
                gov_dep.append(t)

            if train_test=="train":
                gov_dep_all=set()#gov_dep_all=[]
                for e in all_dep_noduplicate:
                    gov_dep_all.add(e[1:])
                gov_dep_all=list(gov_dep_all)

            govdepmat=np.zeros([length,len(gov_dep_all)])
            i=0
            for dep in gov_dep:#each tweet list
                for e in dep:
                    if e in gov_dep_all:

                        ind=gov_dep_all.index(e)
                        govdepmat[i][ind]=1
                i+=1
            features=all_dep_noduplicate+rel_gov_all+gov_dep_all

            def reduce_size(mat,f):
                j = 0;ff=[]
                for i in range(mat.shape[1]):
                    if np.sum(mat, axis=0)[i] >= 2:
                        if j == 0:
                            y = np.vstack(mat[:, i])
                            ff.append(f[i])
                            j = 1
                        else:
                            k = np.vstack(mat[:, i])
                            y = np.hstack((y, k))
                            ff.append(f[i])
                return y,ff
            if train_test=="train":
                relgovdepmat,all_dep_noduplicate=reduce_size(relgovdepmat,all_dep_noduplicate)
                relgovmat,rel_gov_all=reduce_size(relgovmat,rel_gov_all)
                govdepmat,gov_dep_all=reduce_size(govdepmat,gov_dep_all)
                features=all_dep_noduplicate+rel_gov_all+gov_dep_all
                return relgovdepmat,relgovmat,govdepmat,features,all_dep_noduplicate,rel_gov_all,gov_dep_all

            return relgovdepmat,relgovmat,govdepmat,features


        train_depen,alldepen=get_dependency(train,"train")
        test_depen=get_dependency(test,"test")
        relgovdep_train,relgov_train,govdep_train,feat_train,all_dep_noduplicate,rel_gov_all,gov_dep_all=rel_gov_dep(train_depen,alldepen,len(train),'train')
        relgovdep_test,relgov_test,govdep_test,feat_test=rel_gov_dep(test_depen,all_dep_noduplicate,len(test),'test',rel_gov_all,gov_dep_all)

        train_matrix=np.hstack((relgovdep_train,relgov_train,govdep_train))
        test_matrix=np.hstack((relgovdep_test,relgov_test,govdep_test))


        newfeat = []; r = ""
        for each_feat in feat_train:
            r = "stanpars"
            for i in each_feat:
                r = r + "_" + i
            newfeat.append(r)
        feat_train=newfeat

        #print("number of features ", len(feat_train))



        def reducingsize(trainmatrix,testmatrix,feat):
            try:
                TRAIN = np.squeeze(np.asarray(trainmatrix.todense()))
                TEST = np.squeeze(np.asarray(testmatrix.todense()))
            except:
                TRAIN=trainmatrix
                TEST=testmatrix
            c = []
            for e in TRAIN.T:
                c.append(len(np.where(e != 0)[0]))
            ind=[]
            for i in range(len(c)):
                if c[i] >= 3 :
                    ind.append(i)
            ind=np.array(ind)
            trainmatrix=TRAIN[:,ind]
            testmatrix=TEST[:,ind]
            feat = [feat[g] for g in ind]
            return trainmatrix,testmatrix,feat


        if MINDF == 3:
            mincount_stdparser = True
        else:
            mincount_stdparser = False
        if mincount_stdparser==True:
            train_matrix,test_matrix,feat_train=reducingsize(train_matrix,test_matrix,feat_train)

        joblib.dump(train_matrix,trainfile,protocol=2)
        joblib.dump(test_matrix,testfile,protocol=2)

        joblib.dump(feat_train,featfile,protocol=2)

        return csr_matrix(train_matrix),csr_matrix(test_matrix),feat_train


    def browncluster(self,tweets,tweets_train):

        trainfile =self.pathfile+ "train_" + "browncluster" + '.pkl'
        testfile = self.pathfile+"test_" + "browncluster" + '.pkl'
        featfile = self.pathfile+"feat_" + "browncluster" + '.pkl'
        if os.path.isfile(trainfile):
            train_mat = joblib.load(trainfile)
            if os.path.isfile(testfile):
                test_mat= joblib.load(testfile)
                if os.path.isfile(featfile):
                    clusters = joblib.load(featfile)
                    return csr_matrix(train_mat), csr_matrix(test_mat), clusters


        import codecs
        import sys
        #clusters = codecs.open('50mpaths2.txt', 'r', encoding='utf8').read().splitlines()
        clusters = codecs.open('resources/50mpaths2.txt', 'r', encoding='utf8').read().splitlines()
        clusters = [i.split("\t") for i in clusters]
        voca_clusters = {i[1]:i[0] for i in clusters}
        clusters = list(set([i[0] for i in clusters]))
        train_clusters=[]
        train_mat=np.zeros([len(tweets_train),len(clusters)])
        i=0
        for tweet in tweets_train:
            tweet_cluster=[]
            tokens=tweet.tokens_nopunc
            for token in tokens:
                if token in voca_clusters:
                    tweet_cluster.append(voca_clusters[token])
                    ind = clusters.index(voca_clusters[token])
                    train_mat[i][ind] += 1
            train_clusters.append(tweet_cluster)
            i+=1
        test_clusters=[]
        test_mat=np.zeros([len(tweets),len(clusters)])
        j=0
        for tweet in tweets:
            tweet_cluster=[]

            tokens=tweet.tokens_nopunc
            for token in tokens:
                if token in voca_clusters:
                    tweet_cluster.append(voca_clusters[token])
                    ind = clusters.index(voca_clusters[token])
                    test_mat[j][ind] += 1
            test_clusters.append(tweet_cluster)
            j+=1

        #print("num features ",len(clusters))

        joblib.dump(train_mat,trainfile,protocol=2)
        joblib.dump(test_mat,testfile,protocol=2)
        joblib.dump(clusters,featfile,protocol=2)
            
        return csr_matrix(train_mat),csr_matrix(test_mat),clusters



    def twise_pos_lexical(self,tweets,tweets_train,req):

        pos_text_train=None

        pos_train_file = self.pathfile+"pos_train" + '.pkl'
        pos_features_train_file= self.pathfile+"pos_features_train"+ '.pkl'

        pos_test_file= self.pathfile+"pos_test" + '.pkl'
        pos_features_test_file= self.pathfile+"pos_features_test"+ '.pkl'

        different_pos_tags_file=self.pathfile+"different_pos_tags" + '.pkl'

        pos_text_test_file=self.pathfile+"pos_text_test"+".pkl"
        pos_text_train_file=self.pathfile+"pos_text_train"+".pkl"
        if os.path.isfile(pos_train_file):
            pos_train=joblib.load(pos_train_file)
            if os.path.isfile(pos_features_train_file):
                pos_features_train=joblib.load(pos_features_train_file)
            if os.path.isfile(pos_test_file):
                pos_test=joblib.load(pos_test_file)
            if os.path.isfile(pos_features_test_file):
                pos_features_test= joblib.load(pos_features_test_file)
            if os.path.isfile(different_pos_tags_file):
                different_pos_tags= joblib.load(different_pos_tags_file)
            if os.path.isfile(pos_text_train_file):
                pos_text_train= joblib.load(pos_text_train_file)
            if os.path.isfile(pos_text_test_file):
                pos_text_test= joblib.load(pos_text_test_file)

        train = [tweet.text_nopunc.lower() for tweet in tweets_train]
        test = [tweet.text_nopunc.lower() for tweet in tweets]



        if pos_text_train==None:
            import twise
            pos_train, pos_features_train, different_pos_tags, pos_text_train=twise.get_pos_tags(train)
            pos_test, pos_features_test,different_pos_tags, pos_text_test=twise.get_pos_tags(test)
            from collections import Counter
            pos_features_train=[]
            pos_features_test=[]
            for e in pos_train:
                e=Counter(e)
                tags=[]
                for POS in different_pos_tags:
                    try:
                        tags.append(e[POS])
                    except:
                        tags.append(0)
                pos_features_train.append(np.array(tags))
            pos_features_train=np.array(pos_features_train)
            for e in pos_test:
                tags=[]
                e=Counter(e)
                for POS in different_pos_tags:
                    try:
                        tags.append(e[POS])
                    except:
                        tags.append(0)
                pos_features_test.append(np.array(tags))
            pos_features_test=np.array(pos_features_test)



            joblib.dump(pos_train,pos_train_file,protocol=2)
            joblib.dump(pos_features_train,pos_features_train_file,protocol=2)
            joblib.dump(different_pos_tags,different_pos_tags_file,protocol=2)
            joblib.dump(pos_text_train,pos_text_train_file,protocol=2)
            joblib.dump(pos_test,pos_test_file,protocol=2)
            joblib.dump(pos_features_test,pos_features_test_file,protocol=2)
            joblib.dump(pos_text_test,pos_text_test_file,protocol=2)


        train_mat=[]
        test_mat=[]
        all_features=[]
        if "pos" in req:

            train_mat.append(pos_features_train)
            test_mat.append(pos_features_test)
            all_features=all_features+different_pos_tags
                        
        if "nrcemotion" in req:
            nrc_train_file=self.pathfile+"nrc_train_file.pkl"
            nrc_test_file=self.pathfile+"nrc_test_file.pkl"
            if os.path.isfile(nrc_train_file):
                nrc_train=joblib.load(nrc_train_file)
                if os.path.isfile(nrc_test_file):
                    nrc_test = joblib.load(nrc_test_file)
            else:
                import twise
                nrc_train=twise.nrc_emotion(train, pos_train, different_pos_tags, pos_text_train ,tknzr="twise")
                nrc_test=twise.nrc_emotion(test, pos_test, different_pos_tags, pos_text_test ,tknzr="twise")
                joblib.dump(nrc_train,nrc_train_file,protocol=2)
                joblib.dump(nrc_test,nrc_test_file,protocol=2)


            train_mat.append(nrc_train)
            test_mat.append(nrc_test)
            f=[]
            for i in range(nrc_train.shape[1]):
                f.append("tw_nrcemo_"+str(i))
            all_features=all_features+f
            
        if "hashaffneg" in req:
            hash_train_file=self.pathfile+"hash_train_file.pkl"
            hash_test_file=self.pathfile+"hash_test_file.pkl"
            if os.path.isfile(hash_train_file):
                hash_train=joblib.load(hash_train_file)
                if os.path.isfile(hash_test_file):
                    hash_test = joblib.load(hash_test_file)
            else:
                #path2lexicon='lexicons/HashtagSentimentAffLexNegLex/HS-AFFLEX-NEGLEX-unigrams.txt'
                path2lexicon='resources/HashtagSentimentAffLexNegLex/HS-AFFLEX-NEGLEX-unigrams.txt'
                import twise
                hash_train=twise.sent140aff(train, pos_train, different_pos_tags, pos_text_train, path2lexicon,tknzr="twise")
                hash_test=twise.sent140aff(test, pos_test, different_pos_tags, pos_text_test, path2lexicon,tknzr="twise")
                joblib.dump(hash_train,hash_train_file,protocol=2)
                joblib.dump(hash_test,hash_test_file,protocol=2)


            train_mat.append(hash_train)
            test_mat.append(hash_test)
            f=[]
            for i in range(hash_train.shape[1]):
                f.append("tw_hashaffneg_"+str(i))
            all_features=all_features+f
        if "hashaffneg-bi" in req:
            hash_bi_train_file=self.pathfile+"hash_bi_train_file.pkl"
            hash_bi_test_file=self.pathfile+"hash_bi_test_file.pkl"
            if os.path.isfile(hash_bi_train_file):
                hash_bi_train=joblib.load(hash_bi_train_file)
                if os.path.isfile(hash_bi_test_file):
                    hash_bi_test = joblib.load(hash_bi_test_file)
            else:
                import twise
                #path2lexicon='lexicons/HashtagSentimentAffLexNegLex/HS-AFFLEX-NEGLEX-bigrams.txt'
                path2lexicon='resources/HashtagSentimentAffLexNegLex/HS-AFFLEX-NEGLEX-bigrams.txt'
                hash_bi_train=twise.sent140aff_bigrams(train, pos_train, different_pos_tags, pos_text_train, path2lexicon)
                hash_bi_test=twise.sent140aff_bigrams(test, pos_test, different_pos_tags, pos_text_test, path2lexicon)
                joblib.dump(hash_bi_train,hash_bi_train_file,protocol=2)
                joblib.dump(hash_bi_test,hash_bi_test_file,protocol=2)

            train_mat.append(hash_bi_train)
            test_mat.append(hash_bi_test)
            f=[]
            for i in range(hash_bi_train.shape[1]):
                f.append("tw_hashaffneg_bi_"+str(i))
            all_features=all_features+f
        if "sent140affneg" in req:
            sent_train_file=self.pathfile+"sent_train_file.pkl"
            sent_test_file=self.pathfile+"sent_test_file.pkl"
            if os.path.isfile(sent_train_file):
                sent_train=joblib.load(sent_train_file)
                if os.path.isfile(sent_test_file):
                    sent_test = joblib.load(sent_test_file)
            else:
                import twise
                #path2lexicon='lexicons/Sentiment140AffLexNegLex/S140-AFFLEX-NEGLEX-unigrams.txt'
                path2lexicon='resources/Sentiment140AffLexNegLex/S140-AFFLEX-NEGLEX-unigrams.txt'
                sent_train=twise.sent140aff(train, pos_train, different_pos_tags, pos_text_train, path2lexicon,tknzr="twise")
                sent_test=twise.sent140aff(test, pos_test, different_pos_tags, pos_text_test, path2lexicon,tknzr="twise")
                joblib.dump(sent_train,sent_train_file,protocol=2)
                joblib.dump(sent_test,sent_test_file,protocol=2)

            train_mat.append(sent_train)
            test_mat.append(sent_test)
            f=[]
            for i in range(sent_train.shape[1]):
                f.append("tw_sentaffneg_"+str(i))
            all_features=all_features+f
        if "sent140affneg-bi" in req:
            sentbi_train_file=self.pathfile+"sentbi_train_file.pkl"
            sentbi_test_file=self.pathfile+"sentbi_test_file.pkl"
            if os.path.isfile(sentbi_train_file):
                sentbi_train=joblib.load(sentbi_train_file)
                if os.path.isfile(sentbi_test_file):
                    sentbi_test = joblib.load(sentbi_test_file)
            else:
                import twise
                #path2lexicon='lexicons/Sentiment140AffLexNegLex/S140-AFFLEX-NEGLEX-bigrams.txt'
                path2lexicon='resources/Sentiment140AffLexNegLex/S140-AFFLEX-NEGLEX-bigrams.txt'
                sentbi_train=twise.sent140aff_bigrams(train, pos_train, different_pos_tags, pos_text_train, path2lexicon)
                sentbi_test=twise.sent140aff_bigrams(test, pos_test, different_pos_tags, pos_text_test, path2lexicon)
                joblib.dump(sentbi_train,sentbi_train_file,protocol=2)
                joblib.dump(sentbi_test,sentbi_test_file,protocol=2)

            train_mat.append(sentbi_train)
            test_mat.append(sentbi_test)
            f=[]
            for i in range(sentbi_train.shape[1]):
                f.append("tw_sentaffneg_bi_"+str(i))
            all_features=all_features+f
        if "mpqa" in req:
            mpqa_train_file = self.pathfile+"mpqa_train_file.pkl"
            mpqa_test_file = self.pathfile+"mpqa_test_file.pkl"
            if os.path.isfile(mpqa_train_file):
                mpqa_train = joblib.load(mpqa_train_file)
                if os.path.isfile(mpqa_test_file):
                    mpqa_test = joblib.load(mpqa_test_file)
            else:
                train = [tweet.text_nopunc.lower() for tweet in tweets_train]
                test = [tweet.text_nopunc.lower() for tweet in tweets]
                import twise
                mpqa_train=twise.mpqa(train, pos_train, different_pos_tags, pos_text_train,tknzr="twise")
                mpqa_test=twise.mpqa(test, pos_test, different_pos_tags, pos_text_test,tknzr="twise")
                joblib.dump(mpqa_train,mpqa_train_file,protocol=2)
                joblib.dump(mpqa_test,mpqa_test_file,protocol=2)

            train_mat.append(mpqa_train)
            test_mat.append(mpqa_test)
            f=[]
            for i in range(mpqa_train.shape[1]):
                f.append("tw_mpqa_"+str(i))
            all_features=all_features+f
            
        if "bingliu" in req:
            bingliu_train_file =self.pathfile+ "bingliu_train_file.pkl"
            bingliu_test_file = self.pathfile+"bingliu_test_file.pkl"
            if os.path.isfile(bingliu_train_file):
                bl_train = joblib.load(bingliu_train_file)
                if os.path.isfile(bingliu_test_file):
                    bl_test = joblib.load(bingliu_test_file)
            else:
                import twise
                train = [tweet.text_nopunc for tweet in tweets_train]
                test = [tweet.text_nopunc for tweet in tweets]
                bl_train=twise.bing_lius(train, pos_train, different_pos_tags, pos_text_train,tknzr="twise" )
                bl_test=twise.bing_lius(test, pos_test, different_pos_tags, pos_text_test,tknzr="twise" )
                joblib.dump(bl_test,bingliu_test_file,protocol=2)
                joblib.dump(bl_train,bingliu_train_file,protocol=2)

            train_mat.append(bl_train)
            test_mat.append(bl_test)
            f=[]
            for i in range(bl_train.shape[1]):
                f.append("tw_bl_"+str(i))
            all_features=all_features+f
        if "sentiwordnet" in req:
            sentiwordnet_train_file = self.pathfile+"sentiwordnet_train_file.pkl"
            sentiwordnet_test_file = self.pathfile+"sentiwordnet_test_file.pkl"
            if os.path.isfile(sentiwordnet_train_file):
                wordnet_train = joblib.load(sentiwordnet_train_file)
                if os.path.isfile(sentiwordnet_test_file):
                    wordnet_test = joblib.load(sentiwordnet_test_file)
            else:
                import twise
                wordnet_train=twise.get_sentiwordnet(pos_text_train, pos_train)
                wordnet_test=twise.get_sentiwordnet(pos_text_test, pos_test)
                joblib.dump(wordnet_train,sentiwordnet_train_file,protocol=2)
                joblib.dump(wordnet_test,sentiwordnet_test_file,protocol=2)

            train_mat.append(wordnet_train)
            test_mat.append(wordnet_test)
            f=[]
            for i in range(wordnet_train.shape[1]):
                f.append("tw_wordnet_"+str(i))
            all_features=all_features+f

        return csr_matrix(np.hstack(train_mat)),csr_matrix(np.hstack(test_mat)),all_features

    def allcaps(self,tweets,tweets_train):#ecnu_paper
        trainfile =self.pathfile+ "train_" + "allcaps" + '.pkl'
        testfile =self.pathfile+ "test_" + "allcaps" + '.pkl'
        if os.path.isfile(trainfile):
            train = joblib.load(trainfile)
            if os.path.isfile(testfile):
                test = joblib.load(testfile)
                return csr_matrix(train).T, csr_matrix(test).T, ["allcaps"]
        test=[];train=[]

        for tweet in tweets:

            words=tweet.text_nopunc#nltk.word_tokenize(tweet.text)
            test.append(len([word for word in words if word.isupper()]))

        for tweet in tweets_train:

            words=tweet.text_nopunc#nltk.word_tokenize(tweet.text)
            train.append(len([word for word in words if word.isupper()]))
        joblib.dump(train,trainfile,protocol=2)
        joblib.dump(test,testfile,protocol=2)

        return csr_matrix(train).T,csr_matrix(test).T,["allcaps"]
    
    def elongated(self,tweets,tweets_train):#ecnu_paper
        trainfile = self.pathfile+"train_" + "elongated" + '.pkl'
        testfile = self.pathfile+"test_" + "elongated" + '.pkl'
        if os.path.isfile(trainfile):
            train = joblib.load(trainfile)
            if os.path.isfile(testfile):
                test = joblib.load(testfile)
                return csr_matrix(train).T, csr_matrix(test).T, ["elongated"]

        test=[];train=[]


        for tweet in tweets:
            words=tweet.tokens_nopunc
            test.append(len([word for word in words if re.search(r"(.)\1{2}", word.lower())]))

        for tweet in tweets_train:
            words=tweet.tokens_nopunc
            train.append(len([word for word in words if re.search(r"(.)\1{2}", word.lower())]))
        joblib.dump(train,trainfile,protocol=2)
        joblib.dump(test,testfile,protocol=2)

        return csr_matrix(train).T,csr_matrix(test).T,["elongated"]
    
    def get_ngrams_binary(self,tweets,tweets_train,MINDF):
        trainfile = self.pathfile+"train_" + "ngrams-bin" + '.pkl'
        testfile = self.pathfile+"test_" + "ngrams-bin" + '.pkl'
        featfile = self.pathfile+"feat_" + "ngrams-bin" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
            if os.path.isfile(featfile):
                feat = joblib.load(featfile)
                return trainmatrix, testmatrix, feat

        vec = CountVectorizer(ngram_range=(1, 3),lowercase=True, binary=True,min_df=MINDF)

        feat_test = []
        feat_train = []
        for tweet in tweets_train:
            feat_train.append(tweet.text_nopunc)
        for tweet in tweets:
            feat_test.append(tweet.text_nopunc)

        vec.fit_transform(feat_train)
        trainmatrix = vec.transform(feat_train)
        testmatrix = vec.transform(feat_test)
        feat = vec.get_feature_names()

        #print("number of features ", len(feat))

        feat = ["ngramsbin_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix, trainfile, protocol=2)
        joblib.dump(testmatrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)
        return trainmatrix, testmatrix, feat

    def get_hashtag_binary(self, tweets, tweets_train,MINDF):

        trainfile = self.pathfile+"train_" + "hashtag-bin" + '.pkl'
        testfile = self.pathfile+"test_" + "hashtag-bin" + '.pkl'
        featfile = self.pathfile+"feat_" + "hashtag-bin" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat



        feat_train = []
        for t in tweets_train:
            hash = ""
            hashtag = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
            for elem in hashtag:
                elem = elem.lstrip('<hashtag>');
                elem = elem.rstrip('</hashtag>');
                elem = elem.strip()
                hash += elem + " "
            feat_train.append(hash)
        feat_test= []
        for t in tweets:
            hash = ""
            hashtag = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
            for elem in hashtag:
                elem = elem.lstrip('<hashtag>');
                elem = elem.rstrip('</hashtag>');
                elem = elem.strip()
                hash += elem + " "
            feat_test.append(hash)


        countvec = CountVectorizer(ngram_range=(1, 3),binary=True, min_df=MINDF)

        countvec.fit_transform(feat_train)
        trainmatrix = countvec.transform(feat_train)
        testmatrix = countvec.transform(feat_test)
        feat = countvec.get_feature_names()
        #print("number of features ", len(feat))

        feat = ["hashtagbin_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix, trainfile, protocol=2)
        joblib.dump(testmatrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)

        return trainmatrix, testmatrix, feat

    def get_screen_binary(self, tweets, tweets_train,MINDF):
        trainfile = self.pathfile+"train_" + "screen-bin" + '.pkl'
        testfile = self.pathfile+"test_" + "screen-bin" + '.pkl'
        featfile = self.pathfile+"feat_" + "screen-bin" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat

        feat_train = []
        for t in tweets_train:
            hash = ""
            hashtag = re.findall(r'<user>[\s+\w+\._?!]+</user>', t.text_hash)
            for elem in hashtag:
                elem = elem.lstrip('<user>');
                elem = elem.rstrip('</user>');
                elem = elem.strip()
                hash += elem + " "
            feat_train.append(hash)
        feat_test= []
        for t in tweets:
            hash = ""
            hashtag = re.findall(r'<user>[\s+\w+\._?!]+</user>', t.text_hash)
            for elem in hashtag:
                elem = elem.lstrip('<user>');
                elem = elem.rstrip('</user>');
                elem = elem.strip()
                hash += elem + " "
            feat_test.append(hash)


        countvec = CountVectorizer(ngram_range=(1,2),binary=True,min_df=MINDF)
        countvec.fit_transform(feat_train)
        trainmatrix = countvec.transform(feat_train)
        testmatrix = countvec.transform(feat_test)
        feat = countvec.get_feature_names()
        #print("number of features ", len(feat))

        feat = ["screenbin_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix, trainfile, protocol=2)
        joblib.dump(testmatrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)

        return trainmatrix, testmatrix, feat


    def get_BoWinHashtag(self, tweets, tweets_train,against_list,favor_list,target_list,MINDF):

        trainfile = self.pathfile+"train_" + "BoWinHashtag" + '.pkl'
        testfile = self.pathfile+"test_" + "BoWinHashtag" + '.pkl'
        featfile = self.pathfile+"feat_" + "BoWinHashtag" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat



        def get_feature(tweets,target_name,target_pronoums,target_party,target_opposite_in_party,target_opposite_out_party):
            feature = []
            for tweet in tweets:

                text_tag = ""
                hashtag = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', tweet.text_hash.lower())
                for elem in hashtag:
                    elem = elem.lstrip('<hashtag>');
                    elem = elem.rstrip('</hashtag>');
                    elem = elem.strip()
                    text_tag += elem + " "


                text_tag = text_tag.lower()
                for t in target_name:
                    text_tag = text_tag.replace(t.lower(), " feature_bow_target_TARGET ")
                for t in target_pronoums:
                    text_tag = text_tag.replace(t.lower(), " feature_bow_target_TARGET_PRONOUN ")
                for t in target_party:
                    text_tag = text_tag.replace(t.lower(), " feature_bow_target_PARTY ")
                for t in target_opposite_in_party:
                    text_tag = text_tag.replace(t.lower(), " feature_bow_target_IN_PARTY ")
                for t in target_opposite_out_party:
                    text_tag = text_tag.replace(t.lower(), " feature_bow_target_out_party ")
                feature.append(text_tag)
            return feature
        def get_feature_other(tweets):
            feature = []
            for tweet in tweets:
                text_tag = ""
                hashtag = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', tweet.text_hash.lower())
                for elem in hashtag:
                    elem = elem.lstrip('<hashtag>');
                    elem = elem.rstrip('</hashtag>');
                    elem = elem.strip()
                    text_tag += elem + " "
                text_tag = text_tag.lower()

                for t in target_list:#words_target:
                    text_tag = text_tag.replace(t.lower(), " feature_bow_target_TARGET ")
                for t in  favor_list:#words_favor:
                    text_tag = text_tag.replace(t.lower(), " feature_bow_target_FAVOR ")
                for t in against_list:#words_against:
                    text_tag = text_tag.replace(t.lower(), " feature_bow_target_AGAINST ")
                feature.append(text_tag)
            return feature

        feature_test = get_feature_other(tweets)
        feature_train = get_feature_other(tweets_train)


        tfidfVectorizer = CountVectorizer(ngram_range=(1, 3),  lowercase=True,  binary=True,min_df=MINDF)
        tfidfVectorizer = tfidfVectorizer.fit(feature_train)
        trainmatrix = tfidfVectorizer.transform(feature_train)
        testmatrix = tfidfVectorizer.transform(feature_test)
        feat = tfidfVectorizer.get_feature_names()

        #print("number of features ", len(feat))

        feat = ["bowinhashtag_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix, trainfile, protocol=2)
        joblib.dump(testmatrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)

        return  trainmatrix, testmatrix, feat

    def target_binary(self,tweets, tweets_train,against_list,favor_list,target_list,MINDF):


        trainfile = self.pathfile+"train_" + "target-bin "+ '.pkl'
        testfile = self.pathfile+"test_" + "target-bin" + '.pkl'
        featfile = self.pathfile+"feat_" + "target-bin" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat



        def getfeature(tweets,target_name,target_pronoums,target_party,target_opposite_in_party,target_opposite_out_party):
            feature  = []
            for tweet in tweets:
                text=""
                text_tweet=re.sub(r'[^\w\s]',' ',tweet.text_nopunc.lower())
                text_tweet=text_tweet.split(" ")

                for t in target_name:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_bin_xTARGETx "

                for t in target_pronoums:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_bin_xTARGETx "

                for t in target_party:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_bin_xTARGETx "

                for t in target_opposite_in_party:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_bin_xOPPOSITEINPARTYTARGETx "

                for t in target_opposite_out_party:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_bin_xOPPOSITEINPARTYTARGETx "

                if text=="":
                    text+=" feature_target_in_tweet_bin_xNOTHINGx "

                feature.append(text)
            return feature
        def getfeature_other(tweets):
            feature  = []

            for tweet in tweets:
                text=""
                text_tweet=re.sub(r'[^\w\s]',' ',tweet.text_nopunc.lower())
                text_tweet=text_tweet.split(" ")

                for t in target_list:#words_target:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_bin_xTARGETx "

                for t in favor_list:#words_favor:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_bin_xTARGETx "

                for t in against_list:#words_against:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_bin_xAGAINSTx "

                if text=="":
                    text+=" feature_target_in_tweet_bin_xNOTHINGx "

                feature.append(text)
            return feature


        feature = getfeature_other(tweets)
        feature_train = getfeature_other(tweets_train)


        tfidfVectorizer = CountVectorizer(ngram_range=(1,3),
                                          binary=True,
                                          min_df=MINDF)

        tfidfVectorizer = tfidfVectorizer.fit(feature_train)
        trainamtrix = tfidfVectorizer.transform(feature_train)
        testmatrix = tfidfVectorizer.transform(feature)
        feat=tfidfVectorizer.get_feature_names()
        #print("number of features ", len(feat))

        feat = ["targetbin_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainamtrix, trainfile, protocol=2)
        joblib.dump(testmatrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)

        return trainamtrix,testmatrix,feat





    def get_target_in_tweet(self,tweets, tweets_train,against_list,favor_list,target_list,MINDF):


        trainfile = self.pathfile+"train_" + "targetintweet" + '.pkl'
        testfile = self.pathfile+"test_" + "targetintweet" + '.pkl'
        featfile = self.pathfile+"feat_" + "targetintweet" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat



        def getfeature(tweets,target_name, target_pronoums, target_party,
                                       target_opposite_in_party, target_opposite_out_party):
            feature  = []
            for tweet in tweets:
                text=""
                text_tweet=re.sub(r'[^\w\s]',' ',tweet.text_nopunc.lower())
                text_tweet=text_tweet.split(" ")

                for t in target_name:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xTARGETx "

                for t in target_pronoums:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xPRONOUMSx "

                for t in target_party:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xPARTYx "

                for t in target_opposite_in_party:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xOPPOSITEINPARTYTARGETx "

                for t in target_opposite_out_party:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xOPPOSITEOUTPARTYTARGETx "

                if text=="":
                    text+=" feature_target_in_tweet_xNOTHINGx "

                feature.append(text)
            return feature

        def getfeature_other(tweets):
            feature = []
            for tweet in tweets:
                text = ""
                text_tweet = re.sub(r'[^\w\s]', ' ', tweet.text_nopunc.lower())
                text_tweet = text_tweet.split(" ")

                for t in target_list:#words_target:
                    if t.lower() in text_tweet:
                        text += " feature_target_in_tweet_xTARGETx "

                for t in favor_list:#words_favor:
                    if t.lower() in text_tweet:
                        text += " feature_target_in_tweet_xFAVORx "

                for t in against_list:#words_against:
                    if t.lower() in text_tweet:
                        text += " feature_target_in_tweet_xAGAINSTx "

                if text == "":
                    text += " feature_target_in_tweet_xNOTHINGx "

                feature.append(text)
            return feature

        feature = getfeature_other(tweets)
        feature_train = getfeature_other(tweets_train)


        tfidfVectorizer = CountVectorizer(ngram_range=(1,3),
                                          binary=False,
                                          min_df=MINDF)
        tfidfVectorizer = tfidfVectorizer.fit(feature_train)
        trainmatrix = tfidfVectorizer.transform(feature_train)
        testmatrix = tfidfVectorizer.transform(feature)
        feat=tfidfVectorizer.get_feature_names()

        #print("number of features ", len(feat))

        feat = ["targetintweet_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix, trainfile, protocol=2)
        joblib.dump(testmatrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)

        return trainmatrix,testmatrix,feat


    def get_target_and_no_in_hashtag(self, tweets, tweets_train,against_list,favor_list,target_list,MINDF):


        trainfile = self.pathfile+"train_" + "targetandnoinhashtag" + '.pkl'
        testfile = self.pathfile+"test_" + "targetandnoinhashtag" + '.pkl'
        featfile = self.pathfile+"feat_" + "targetandnoinhashtag" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat = joblib.load(featfile)
                    return trainmatrix, testmatrix, feat



        def targetandno(tweets,target_name, target_pronoums, target_party,
                                 target_opposite_in_party, target_opposite_out_party):
            feature = []
            for tweet in tweets:
                text_tag = []
                hashtag = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', tweet.text_hash.lower())
                for elem in hashtag:
                    elem = elem.lstrip('<hashtag>');
                    elem = elem.rstrip('</hashtag>');
                    elem = elem.strip()
                    text_tag.append(elem)

                text = ""
                for tag in text_tag:
                    if "no" in tag or "against" in tag or "stop" in tag or "anti" or "not" in tag:
                        for t in target_name:
                            if t.lower() in tag:
                                text += " feature_target_and_no_in_hashtag_xTARGETx "

                        for t in target_opposite_in_party:
                            if t.lower() in tag:
                                text += " feature_target_and_no_in_hashtag_xOPPOSITETARGETx "
                feature.append(text)
            return feature
        def targetandno_other(tweets):
            feature = []
            for tweet in tweets:
                text_tag = []
                hashtag = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', tweet.text_hash.lower())
                for elem in hashtag:
                    elem = elem.lstrip('<hashtag>');
                    elem = elem.rstrip('</hashtag>');
                    elem = elem.strip()
                    text_tag.append(elem)

                text = ""
                for tag in text_tag:
                    if "no" in tag or "against" in tag or "stop" in tag or "anti" or "not" in tag:

                        thislist=list(target_list)+list(favor_list)
                        thislist=set(thislist)
                        for t in thislist:#words_target:
                            if t.lower() in tag:
                                text += " feature_target_and_no_in_hashtag_xTARGETx "
                        for t in against_list:
                            if t.lower() in tag:
                                text += " feature_target_and_no_in_hashtag_xOPPOSITETARGETx "
                feature.append(text)
            return feature

        feature = targetandno_other(tweets)
        feature_train = targetandno_other(tweets_train)

        tfidfVectorizer = CountVectorizer(ngram_range=(1, 3),binary=True, min_df=MINDF)

        tfidfVectorizer = tfidfVectorizer.fit(feature_train)
        trainmatrix = tfidfVectorizer.transform(feature_train)
        testmatrix = tfidfVectorizer.transform(feature)
        feat = tfidfVectorizer.get_feature_names()

        feat = ["targetandnoinhashtag_" + re.sub(" ", "_", e) for e in feat]
        joblib.dump(trainmatrix, trainfile, protocol=2)
        joblib.dump(testmatrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)
        return trainmatrix, testmatrix, feat



    def weighted_mpqa_new(self,tweets,tweets_train):
        trainfile = self.pathfile+"train_" + "mpqa_new" + '.pkl'
        testfile = self.pathfile+"test_" + "mpqa_new" + '.pkl'
        featfile = self.pathfile+"feat_" + "mpqa_new" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat = joblib.load(featfile)
                    return csr_matrix(trainmatrix), csr_matrix(testmatrix), feat

        #voca = codecs.open('lexicons/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff','r').read().splitlines()
        voca = codecs.open('resources/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff','r').read().splitlines()

        wds1, wds = {}, {}
        direction = {'negative': -1, 'positive': 1, 'neutral': 0, 'both': 0, 'weaksubj': 1, 'strongsubj': 2}

        for i in voca:
            i = i.split()
            if i[5].split('=')[1] in ['positive', 'negative']:
                wds1[i[2].split('=')[1]] = i[5].split('=')[1]#
                wds[i[2].split('=')[1]]=(i[0].split('=')[1], i[5].split('=')[1])
        tweets_train_matrix=[]
        tweets_matrix=[]
        for tweet in tweets_train:
            pp, pn = 0, 0

            TweeT=tweet.tokens_nopunc

            for i in TweeT:
                try:
                    if i.lower() in wds:
                        if direction[wds[i.lower()][1]] > 0:
                            pp += direction[wds[i.lower()][0]]*direction[wds[i.lower()][1]]
                        if direction[wds[i][1]] < 0:
                            pn += direction[wds[i.lower()][0]]*direction[wds[i.lower()][1]]#
                except:pass
            if(pn+pp)> 2 or (pn+pp)< -2:
                tweets_train_matrix.append(1)
            else:
                tweets_train_matrix.append(0)
        for tweet in tweets:
            pp, pn = 0, 0

            TweeT = tweet.tokens_nopunc
            for i in TweeT:
                try:
                    if i.lower() in wds:
                        if direction[wds[i.lower()][1]] > 0:
                            pp += direction[wds[i][0]]*direction[wds[i.lower()][1]]
                        if direction[wds[i.lower()][1]] < 0:
                            pn += direction[wds[i.lower()][0]]*direction[wds[i.lower()][1]]
                except:pass
            if(pn+pp)> 2 or (pn+pp)< -2:
                tweets_matrix.append(1)#
            else:
                tweets_matrix.append(0)

        tweets_train_matrix=csr_matrix(tweets_train_matrix).T
        tweets_matrix=csr_matrix(tweets_matrix).T
        feat=['mpqa_new']

        joblib.dump(tweets_train_matrix, trainfile, protocol=2)
        joblib.dump(tweets_matrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)
        return tweets_train_matrix, tweets_matrix, feat

    def polar_mpqa_new(self,tweets,tweets_train):
        trainfile = self.pathfile+"train_" + "polar_mpqa_new" + '.pkl'
        testfile = self.pathfile+"test_" + "polar_mpqa_new" + '.pkl'
        featfile = self.pathfile+"feat_" + "polar_mpqa_new" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat = joblib.load(featfile)
                    return csr_matrix(trainmatrix), csr_matrix(testmatrix), feat

        #voca = codecs.open('lexicons/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff','r').read().splitlines()
        voca = codecs.open('resources/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff','r').read().splitlines()

        wds1, wds = {}, {}
        direction = {'negative': -1, 'positive': 1, 'neutral': 0, 'both': 0, 'weaksubj': 1, 'strongsubj': 2}

        for i in voca:
            i = i.split()#
            if i[5].split('=')[1] in ['positive', 'negative']:
                wds1[i[2].split('=')[1]] = i[5].split('=')[1]#
        tweets_train_matrix=[]
        tweets_matrix=[]
        for tweet in tweets_train:
            score=0
            for token in tweet.tokens_nopunc:#
                try:
                    if token.lower() in wds1:
                        score+=direction[wds1[token.lower()]]
                except:pass
            tweets_train_matrix.append(score)
        for tweet in tweets:
            score = 0
            for token in tweet.tokens_nopunc:#
                try:
                    if token.lower() in wds1:
                        score+=direction[wds1[token.lower()]]
                except:pass
            tweets_matrix.append(score)

        tweets_train_matrix=csr_matrix(tweets_train_matrix).T
        tweets_matrix=csr_matrix(tweets_matrix).T
        feat=['polar_mpqa_new']

        joblib.dump(tweets_train_matrix, trainfile, protocol=2)
        joblib.dump(tweets_matrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)

        return tweets_train_matrix, tweets_matrix, feat


    def wordnet_new(self,tweets,tweets_train):
        #http://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html
        #http://www.nltk.org/howto/wordnet.html
        #https://stackoverflow.com/questions/35462747/how-to-check-a-word-if-it-is-adjective-or-verb-using-python-nltk
        #https://groups.google.com/forum/#!topic/nltk-users/tQdUXTQj1gQ
        #http://wordnet.princeton.edu/wordnet/man/wngloss.7WN.html
        trainfile = self.pathfile+"train_" + "wordnet_new" + '.pkl'
        testfile = self.pathfile+"test_" + "wordnet_new" + '.pkl'
        featfile = self.pathfile+"feat_" + "wordnet_new" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat = joblib.load(featfile)
                    return csr_matrix(trainmatrix), csr_matrix(testmatrix), feat

        import nltk
        from nltk.corpus import wordnet
        train_matrix=[]
        test_matrix=[]
        for tweet in tweets:
            feat_test = 0
            for token in tweet.tokens_nopunc:#
                try:
                    syns = wordnet.synsets(token.lower())
                    if syns[0].pos() == 'a' or syns[0].pos() == 's' :
                        feat_test=1
                except:pass
            test_matrix.append(feat_test)

        for tweet in tweets_train:
            feat_train = 0;
            for token in tweet.tokens_nopunc:#
                try:
                    syns = wordnet.synsets(token.lower())
                    if syns[0].pos() == 'a' or syns[0].pos() == 's' :
                        feat_train=1
                except:pass
            train_matrix.append(feat_train)

        tweets_train_matrix = csr_matrix(train_matrix).T
        tweets_matrix = csr_matrix(test_matrix).T
        feat = ['wordnet_new']

        joblib.dump(tweets_train_matrix, trainfile, protocol=2)
        joblib.dump(tweets_matrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)

        return tweets_train_matrix, tweets_matrix, feat

    def sentiwordnet_new(self,tweets,tweets_train):
        trainfile = self.pathfile+"train_" + "sentiwn_new" + '.pkl'
        testfile = self.pathfile+"test_" + "sentiwn_new" + '.pkl'
        featfile = self.pathfile+"feat_" + "sentiwn_new" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
                if os.path.isfile(featfile):
                    feat = joblib.load(featfile)
                    return csr_matrix(trainmatrix), csr_matrix(testmatrix), feat
        import nltk
        from nltk.corpus import sentiwordnet as swn
        from nltk.corpus import wordnet
        train_matrix=[]
        test_matrix=[]
        for tweet in tweets:
            score=0
            for token in tweet.tokens_nopunc:#
                try:
                    syns = wordnet.synsets(token.lower())
                    ##
                    s = swn.senti_synset(syns[0].name())
                    if (s.pos_score() > 0):
                        score += 1
                    if (s.neg_score() > 0):
                        score -= 1



                except:
                    pass
            test_matrix.append(score)

        for tweet in tweets_train:
            score=0
            for token in tweet.tokens_nopunc:#
                try:
                    syns = wordnet.synsets(token.lower())
                    s = swn.senti_synset(syns[0].name())
                    if (s.pos_score() > 0):
                        score += 1
                    if (s.neg_score() > 0):
                        score -= 1

                except:pass

            train_matrix.append(score)

        tweets_train_matrix = csr_matrix(train_matrix).T
        tweets_matrix = csr_matrix(test_matrix).T
        feat = ['sentiwn_new']

        joblib.dump(tweets_train_matrix, trainfile, protocol=2)
        joblib.dump(tweets_matrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)
        return tweets_train_matrix, tweets_matrix, feat

    def ngrams_target_new(self, tweets, tweets_train):
        trainfile = self.pathfile + "train_" + "ngrams_target_new" + '.pkl'
        testfile = self.pathfile + "test_" + "ngrams_target_new" + '.pkl'
        featfile = self.pathfile + "feat_" + "ngrams_target_new" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
            if os.path.isfile(featfile):
                feat = joblib.load(featfile)
                return trainmatrix, testmatrix, feat

        def find_grams(tweet):
            one_gram=0;two_gram=0;three_gram=0;
            if self.TARGET=="Hillary Clinton":
                if len(re.findall('Hilary|Clinton|Hillary', tweet.text_nopunc, flags=re.IGNORECASE))>0: one_gram=1
                if len(re.findall('Hilary Clinton|Hillary Clinton', tweet.text_nopunc, flags=re.IGNORECASE))>0: two_gram=1
                three_gram=0
            elif self.TARGET=="Atheism":
                if len(re.findall('Atheism',  tweet.text_nopunc, flags=re.IGNORECASE))>0: one_gram=1
                two_gram=0;three_gram=0
            elif self.TARGET=="Feminist Movement":
                if len(re.findall('Feminist|Movement',  tweet.text_nopunc, flags=re.IGNORECASE))>0:one_gram=1
                if len(re.findall('Feminist Movement',  tweet.text_nopunc, flags=re.IGNORECASE))>0:two_gram=1
                three_gram=0
            elif self.TARGET=="Legalization of Abortion":
                if len(re.findall('Legalization|Abortion', tweet.text_nopunc, flags=re.IGNORECASE))>0:one_gram=1
                if len(re.findall('Legalization of Abortion',  tweet.text_nopunc, flags=re.IGNORECASE))>0:two_gram=1
                three_gram=0
            elif self.TARGET=="Climate Change is a Real Concern":
                if len(re.findall('Climate|Change|Real|Concern',  tweet.text_nopunc, flags=re.IGNORECASE))>0:one_gram=1
                if len(re.findall('Climate Change|change is|real Concern',  tweet.text_nopunc, flags=re.IGNORECASE))>0:two_gram=1
                if len(re.findall('Climate Change is|change is a|is a Real|a real Concern|climate change real|climate real concern|climate change concern',  tweet.text_nopunc, flags=re.IGNORECASE))>0:three_gram=1
            if one_gram==1 or two_gram==1 or three_gram==1:
                anygram=1
            else: anygram=0
            return one_gram , two_gram,three_gram,anygram

        test_one_grams=[];test_two_grams=[];test_three_grams=[];test_anygram=[]
        for tweet in tweets:
            one_gram,two_gram,three_gram,anygram=find_grams(tweet)
            test_one_grams.append(one_gram)
            test_two_grams.append(two_gram)
            test_three_grams.append(three_gram)
            test_anygram.append(anygram)
        train_one_grams = [];train_two_grams = [];train_three_grams = [];train_anygram=[]
        for tweet in tweets_train:
            one_gram,two_gram,three_gram,anygram=find_grams(tweet)
            train_one_grams.append(one_gram)
            train_two_grams.append(two_gram)
            train_three_grams.append(three_gram)
            train_anygram.append(anygram)
        testmatrix=csr_matrix(np.vstack( (test_one_grams, test_two_grams, test_three_grams, test_anygram))).T
        trainmatrix = csr_matrix(np.vstack((train_one_grams, train_two_grams, train_three_grams, train_anygram))).T
        feat=['one_gram_new',"two_gram_new","three_gram_new","anygram_new"]

        joblib.dump(trainmatrix, trainfile, protocol=2)
        joblib.dump(testmatrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)
        return trainmatrix, testmatrix, feat  #####

    def cgrams_target_new(self, tweets, tweets_train):
        trainfile = self.pathfile + "train_" + "cgrams_target_new" + '.pkl'
        testfile = self.pathfile + "test_" + "cgrams_target_new" + '.pkl'
        featfile = self.pathfile + "feat_" + "cgrams_target_new" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
            if os.path.isfile(featfile):
                feat = joblib.load(featfile)
                return trainmatrix, testmatrix, feat

        def find_grams(tweet):
            four_gram = 0;
            two_gram = 0;
            three_gram = 0;
            five_gram=0;
            if self.TARGET == "Hillary Clinton":
                if len(re.findall('Hi|il|ll|la|ar|ry|Cl|li|in|nt|to|on',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: two_gram = 1
                if len(
                    re.findall('Hil|ill|lla|lar|ary|ila|Cli|lin|int|nto|ton',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: three_gram = 1
                if len(
                    re.findall('Hill|illa|llar|lary|Clin|lint|into|nton|hila|ilar',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: four_gram = 1
                if len(
                    re.findall('Hilla|illar|llary|Clint|linto|inton|hilar|ilary',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: five_gram = 1
            elif self.TARGET == "Atheism":
                if len(re.findall('At|th|he|ei|is|sm',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: two_gram = 1
                if len(re.findall('Ath|the|hei|eis|ism',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: three_gram = 1
                if len(re.findall('Athe|thei|heis|eism',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: four_gram = 1
                if len(re.findall('Athei|theis|heism',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: five_gram = 1

            elif self.TARGET == "Feminist Movement":
                if len(re.findall('Fe|em|mi|in|ni|is|st|mo|ov|ve|em|me|en|nt',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: two_gram = 1
                if len(re.findall('Fem|emi|min|ini|nis|ist|mov|ove|vem|eme|men|ent',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: three_gram = 1
                if len(re.findall('Femi|emin|mini|inis|nist|move|ovem|veme|emen|ment',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: four_gram = 1
                if len(re.findall('Femin|emini|minis|inist|movem|oveme|vemen|ement',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: five_gram = 1

            elif self.TARGET == "Legalization of Abortion":
                if len(re.findall('Le|eg|ga|al|li|iz|za|at|ti|io|on|Ab|bo|or|rt|ti|io|on',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: two_gram = 1
                if len(re.findall('Leg|ega|gal|ali|liz|iza|zat|ati|tio|ion|Abo|bor|ort|rti|tio|ion',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: three_gram = 1
                if len(re.findall('Lega|egal|gali|aliz|liza|izat|zati|atio|tion|Abor|bort|orti|rtio|tion',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: four_gram = 1
                if len(re.findall('Legal|egali|galiz|aliza|lizat|izati|zatio|ation|Abort|borti|ortio|rtion',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: five_gram = 1

            elif self.TARGET == "Climate Change is a Real Concern":
                if len(re.findall('Cl|li|im|ma|at|te|Ch|ha|an|ng|ge|Re|ea|al|Co|on|nc|ce|er|rn',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: two_gram=1
                if len(re.findall('Cli|lim|ima|mat|ate|Cha|han|ang|nge|Rea|eal|Con|onc|nce|cer|ern',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: three_gram=1
                if len(re.findall('Clim|lima|imat|mate|Chan|hang|ange|Real|Conc|once|ncer|cern',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: four_gram=1
                if len(re.findall('Clima|limat|imate|Chang|hange|Conce|oncer|ncern',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: five_gram=1
            if two_gram==1 or three_gram==1 or four_gram==1 or five_gram==1:
                anygram=1;
            else : anygram=0

            return two_gram, three_gram,four_gram,five_gram,anygram

        test_two_grams = [];
        test_three_grams = [];        test_four_grams = [];test_five_grams = [];test_any_grams=[]
        for tweet in tweets:
            two_gram, three_gram ,four_gram,five_gram,any_gram= find_grams(tweet)
            test_two_grams.append(two_gram)
            test_three_grams.append(three_gram)
            test_four_grams.append(four_gram)
            test_five_grams.append(five_gram)
            test_any_grams.append(any_gram)

        train_two_grams = [];
        train_three_grams = [];train_four_grams = [];train_five_grams = [];train_any_grams = [];
        for tweet in tweets_train:
            two_gram, three_gram,four_gram ,five_gram,any_gram= find_grams(tweet)
            train_two_grams.append(two_gram)
            train_three_grams.append(three_gram)
            train_four_grams.append(four_gram)
            train_five_grams.append(five_gram)
            train_any_grams.append(any_gram)


        testmatrix = csr_matrix(np.vstack((test_two_grams, test_three_grams,test_four_grams,test_five_grams,test_any_grams))).T
        trainmatrix = csr_matrix(np.vstack( (train_two_grams, train_three_grams,train_four_grams,train_five_grams,train_any_grams))).T
        feat = ["two_cgram_new", "three_cgram_new",'four_cgram_new','five_cgram_new','any_cgram_new']

        joblib.dump(trainmatrix, trainfile, protocol=2)
        joblib.dump(testmatrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)
        return trainmatrix, testmatrix, feat  #####

    def target_detection_new(self,tweets,tweets_train):
        trainfile = self.pathfile + "train_" + "target_detect_new" + '.pkl'
        testfile = self.pathfile + "test_" + "target_detect_new" + '.pkl'
        featfile = self.pathfile + "feat_" + "target_detect_new" + '.pkl'

        if os.path.isfile(trainfile):
            trainmatrix = joblib.load(trainfile)
            if os.path.isfile(testfile):
                testmatrix = joblib.load(testfile)
            if os.path.isfile(featfile):
                feat = joblib.load(featfile)
                return csr_matrix(np.vstack(trainmatrix)), csr_matrix(np.vstack(testmatrix)), feat

        def find_target(tweet):
            target_detect= 0
            if self.TARGET == "Hillary Clinton":
                if len(re.findall('Hillary Clinton|Hilary Clinton',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: target_detect = 1
            elif self.TARGET == "Atheism":
                if len(re.findall('Atheism',  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: target_detect = 1
            elif self.TARGET == "Feminist Movement":
                if len(re.findall("Feminist Movement", tweet.text_nopunc, flags=re.IGNORECASE)) > 0: target_detect= 1
            elif self.TARGET == "Legalization of Abortion":
                if len(re.findall("Legalization of Abortion",  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: target_detect = 1
            elif self.TARGET == "Climate Change is a Real Concern":
                if len(re.findall("Climate Change is a Real Concern",  tweet.text_nopunc, flags=re.IGNORECASE)) > 0: target_detect=1
            return target_detect
        trainmatrix=[]
        testmatrix=[]
        for tweet in tweets:
            test_target=find_target(tweet)
            testmatrix.append(test_target)
        for tweet in tweets_train:
            train_target=find_target(tweet)
            trainmatrix.append(train_target)
        feat=['target_detect_new']

        joblib.dump(trainmatrix, trainfile, protocol=2)
        joblib.dump(testmatrix, testfile, protocol=2)
        joblib.dump(feat, featfile, protocol=2)

        return csr_matrix(np.vstack(trainmatrix)), csr_matrix(np.vstack(testmatrix)), feat  #####


    #def create_feature_space(self,MINDF,against_list,favor_list,target_list,Google_w2v,twitter_w2v,generate_allfeat,w2v_features,w2v_min_wc,w2v_ctxt,down_sampling,skip_cbow,tweets,tweets_train,target,featureset, favor_words, against_words,favor_tags,against_tags,topn,n_topics,n_passes,first=None):
    def create_feature_space(self, MINDF, against_list, favor_list, target_list, Google_w2v, twitter_w2v,
                                 generate_allfeat, w2v_features, w2v_min_wc, w2v_ctxt, down_sampling, skip_cbow, tweets,
                                 tweets_train, target, featureset, favor_words, against_words, favor_tags, against_tags,first=None):

        if(first==1):
            features = {
                'sentiment_labels': self.get_sentiment(tweets, tweets_train),
                "opinion_labels": self.get_opinion(tweets, tweets_train),
                'word2vecmodel': self.get_word2vec(tweets, tweets_train,w2v_features,w2v_min_wc,w2v_ctxt,down_sampling,skip_cbow,Google_w2v,twitter_w2v),
                'ngrams': self.get_ngrams(tweets, tweets_train,MINDF),
                'ngrams-idf':self.get_ngrams_idf(tweets, tweets_train,MINDF),
                'chars': self.get_char(tweets, tweets_train, 5,MINDF),
                "hashtag": self.get_hashtag(tweets, tweets_train,MINDF),
                "hashtag-idf":self.get_hashtag_idf(tweets, tweets_train,MINDF),
                "screen": self.get_screen(tweets, tweets_train,MINDF),
                "num_hashtag": self.get_num_hashtag(tweets, tweets_train),
                "num_screen": self.get_num_screen(tweets, tweets_train),
                "bowtags": self.bow_hashtag(tweets, tweets_train,against_list,favor_list,target_list,MINDF),
                "bowmentions": self.bow_mention(tweets, tweets_train,against_list,favor_list,target_list,MINDF),
                "posfeat": self.pos_feat(tweets, tweets_train,MINDF),
                "posfeat1": self.pos_feat_1(tweets, tweets_train,MINDF),
                "posfeat2": self.pos_feat_2(tweets, tweets_train,MINDF),
                "sentdal": self.get_dal(tweets, tweets_train),
                "sentafinn": self.get_afinn(tweets, tweets_train),
                "gi":self.get_gi(tweets,tweets_train),
                'tweet_len':self.tweet_len(tweets, tweets_train),
                "avg_word_len":self.avg_word_len(tweets,tweets_train),
                "seq_vowel":self.seq_vowel(tweets,tweets_train),
                "capital_words":self.find_capital_words(tweets,tweets_train),
                "num_punc":self.num_punc(tweets,tweets_train),
                "find_neg":self.find_neg(tweets,tweets_train),
                'stan_parser':self.stan_parser(tweets,tweets_train,MINDF),
                "browncluster":self.browncluster(tweets,tweets_train),
                "twise-pos":self.twise_pos_lexical(tweets,tweets_train,req=["pos"]),
                "twise-hashaffneg":self.twise_pos_lexical(tweets,tweets_train,req=["hashaffneg"]),
                "twise-hashaffneg-bi":self.twise_pos_lexical(tweets,tweets_train,req=["hashaffneg-bi"]),
                "twise-sent140affneg":self.twise_pos_lexical(tweets,tweets_train,req=["sent140affneg"]),
                "twise-sent140affneg-bi":self.twise_pos_lexical(tweets,tweets_train,req=["sent140affneg-bi"]),
                "twise-mpqa":self.twise_pos_lexical(tweets,tweets_train,req=["mpqa"]),
                "twise-bingliu":self.twise_pos_lexical(tweets,tweets_train,req=["bingliu"]),
                "twise-sentiwordnet":self.twise_pos_lexical(tweets,tweets_train,req=["sentiwordnet"]),
                "mpqa":self.mpqa(tweets,tweets_train),
                "bing_lius":self.bing_lius(tweets,tweets_train),
                "NRC_hashtag":self.NRC_hashtag(tweets,tweets_train),
                "NRC_sent140":self.NRC_sent140(tweets,tweets_train),
                "allcaps":self.allcaps(tweets,tweets_train),
                "elongated":self.elongated(tweets,tweets_train),
                "ngrams-bin":self.get_ngrams_binary(tweets,tweets_train,MINDF),
                "hashtag-bin":self.get_hashtag_binary(tweets,tweets_train,MINDF),
                "screen-bin":self.get_screen_binary(tweets,tweets_train,MINDF),
                "BoWinHashtag":self.get_BoWinHashtag(tweets,tweets_train,against_list,favor_list,target_list,MINDF),
                "targetandnoinhashtag":self.get_target_and_no_in_hashtag(tweets,tweets_train,against_list,favor_list,target_list,MINDF),
                "target-bin":self.target_binary(tweets,tweets_train,against_list,favor_list,target_list,MINDF),
                "targetintweet": self.get_target_in_tweet(tweets, tweets_train,against_list,favor_list,target_list,MINDF),
                "mpqa_new":self.weighted_mpqa_new(tweets,tweets_train),
                "wordnet_new":self.wordnet_new(tweets,tweets_train),
                "sentiwn_new":self.sentiwordnet_new(tweets,tweets_train),
                "polar_mpqa_new":self.polar_mpqa_new(tweets,tweets_train),
                "ngrams_target_new":self.ngrams_target_new(tweets,tweets_train),
                "cgrams_target_new":self.cgrams_target_new(tweets,tweets_train),
                "target_detection_new":self.target_detection_new(tweets,tweets_train),

            }


            for key in features:
                file_name = target+"_results/"+target+"_"+key+".pkl"
                pkl_file = open(file_name, 'wb')
                pickle.dump(features[key], pkl_file,protocol=2)
                pkl_file.close()
        
        elif(first==0):
            all_feature_names=[]
            all_X=[]
            all_Y=[]
            for key in featureset:
                X, Y, feature_names = pickle.load(open(target+"_results/"+target+"_" + key + ".pkl", 'rb'))

                all_feature_names=np.concatenate((all_feature_names,feature_names))
                if all_X!=[]:
                    all_X=csr_matrix(hstack((all_X,X)))
                    #https://stackoverflow.com/questions/39388902/python-concatenating-scipy-sparse-matrix
                    all_Y=csr_matrix(hstack((all_Y,Y)))
                else:
                    all_X=X 
                    all_Y=Y 
            return all_X,all_Y,all_feature_names
        elif(first==2):
            def globalfeatures(key):
                #print('key',key)
                if key=='sentiment_labels':return self.get_sentiment(tweets, tweets_train)
                elif key=="opinion_labels":return self.get_opinion(tweets, tweets_train)
                elif key=='word2vecmodel':return self.get_word2vec(tweets, tweets_train,w2v_features,w2v_min_wc,w2v_ctxt,down_sampling,skip_cbow,Google_w2v,twitter_w2v)
                elif key=='ngrams':return self.get_ngrams(tweets, tweets_train,MINDF)
                elif key=='chars':return self.get_char(tweets, tweets_train, 5,MINDF)
                elif key=="hashtag":return self.get_hashtag(tweets, tweets_train,MINDF)
                elif key=="hashtag-idf": return self.get_hashtag_idf(tweets, tweets_train,MINDF)
                elif key=="screen": return self.get_screen(tweets, tweets_train,MINDF)
                elif key=="num_hashtag":return self.get_num_hashtag(tweets, tweets_train)
                elif key=="num_screen": return self.get_num_screen(tweets, tweets_train)
                elif key=="bowtags": return self.bow_hashtag(tweets, tweets_train,against_list,favor_list,target_list,MINDF)
                elif key=="bowmentions": return self.bow_mention(tweets, tweets_train,against_list,favor_list,target_list,MINDF)
                elif key=="posfeat": return self.pos_feat(tweets, tweets_train,MINDF)
                elif key=="posfeat1": return self.pos_feat_1(tweets, tweets_train,MINDF)
                elif key=="posfeat2": return self.pos_feat_2(tweets, tweets_train,MINDF)
                elif key=="sentdal": return self.get_dal(tweets, tweets_train)
                elif key=="sentafinn": return self.get_afinn(tweets, tweets_train)
                elif key=="gi":return self.get_gi(tweets,tweets_train)
                elif key=='ngrams-idf': return self.get_ngrams_idf(tweets, tweets_train,MINDF)
                elif key=='tweet_len': return self.tweet_len(tweets, tweets_train)
                elif key=="avg_word_len": return self.avg_word_len(tweets,tweets_train)
                elif key=="seq_vowel": return self.seq_vowel(tweets,tweets_train)
                elif key=="capital_words":return self.find_capital_words(tweets,tweets_train)
                elif key=="num_punc": return self.num_punc(tweets,tweets_train)
                elif key=="find_neg":return self.find_neg(tweets,tweets_train)
                elif key=="stan_parser":return self.stan_parser(tweets,tweets_train,MINDF)
                elif key=="browncluster":return self.browncluster(tweets,tweets_train)
                elif key=="twise-pos":return self.twise_pos_lexical(tweets,tweets_train,req=["pos"])
                elif key=="twise-hashaffneg":return self.twise_pos_lexical(tweets,tweets_train,req=["hashaffneg"])
                elif key=="twise-hashaffneg-bi":return self.twise_pos_lexical(tweets,tweets_train,req=["hashaffneg-bi"])
                elif key=="twise-sent140affneg":return self.twise_pos_lexical(tweets,tweets_train,req=["sent140affneg"])
                elif key=="twise-sent140affneg-bi":return self.twise_pos_lexical(tweets,tweets_train,req=["sent140affneg-bi"])
                elif key=="twise-mpqa":return self.twise_pos_lexical(tweets,tweets_train,req=["mpqa"])
                elif key=="twise-bingliu":return self.twise_pos_lexical(tweets,tweets_train,req=["bingliu"])
                elif key=="twise-sentiwordnet":return self.twise_pos_lexical(tweets,tweets_train,req=["sentiwordnet"])
                elif key=="twise-all":return self.twise_pos_lexical(tweets,tweets_train,req=["pos","nrcemotion","hashaffneg","hashaffneg-bi","sent140affneg","sent140affneg-bi","mpqa","bingliu","sentiwordnet"])
                elif key=="mpqa":return self.mpqa(tweets,tweets_train)#ecnu_paper
                elif key=="bing_lius":return self.bing_lius(tweets,tweets_train)#ecnu_paper
                elif key=="NRC_hashtag":return self.NRC_hashtag(tweets,tweets_train)#ecnu_paper
                elif key=="NRC_sent140":return self.NRC_sent140(tweets,tweets_train)#ecnu_paper
                elif key=="allcaps":return self.allcaps(tweets,tweets_train)
                elif key=="elongated":return self.elongated(tweets,tweets_train)
                elif key=="ngrams-bin":return self.get_ngrams_binary(tweets,tweets_train,MINDF)
                elif key=="hashtag-bin": return self.get_hashtag_binary(tweets,tweets_train,MINDF)
                elif key=="screen-bin":return self.get_screen_binary(tweets,tweets_train,MINDF)
                elif key=="BoWinHashtag": return self.get_BoWinHashtag(tweets,tweets_train,against_list,favor_list,target_list,MINDF)
                elif key=="targetandnoinhashtag":return self.get_target_and_no_in_hashtag(tweets, tweets_train,against_list,favor_list,target_list,MINDF)
                elif key=="targetintweet": return self.get_target_in_tweet(tweets, tweets_train,against_list,favor_list,target_list,MINDF)
                elif key=="target-bin":return self.target_binary(tweets,tweets_train,against_list,favor_list,target_list,MINDF)
                elif key=="mpqa_new":return self.weighted_mpqa_new(tweets,tweets_train)
                elif key=="wordnet_new": return self.wordnet_new(tweets,tweets_train)
                elif key=="sentiwn_new": return self.sentiwordnet_new(tweets,tweets_train)
                elif key=="polar_mpqa_new": return self.polar_mpqa_new(tweets,tweets_train)
                elif key=="ngrams_target_new": return self.ngrams_target_new(tweets,tweets_train)
                elif key=="cgrams_target_new": return self.cgrams_target_new(tweets,tweets_train)
                elif key=="target_detection_new":return self.target_detection_new(tweets,tweets_train)

            all_feature_names = []
            all_X = []
            all_Y = []
            for key in featureset:
                #print(key)#
                if key.find('$')!=-1:
                    L=key.split('$')
                    feat_index=int(L[1])
                    xx, yy, feature_names = globalfeatures(L[0])
                    try:
                        X=xx.toarray();Y=yy.toarray()
                    except:
                        X=xx;Y=yy
                    X=csr_matrix(np.vstack(X[:,feat_index]));
                    Y=csr_matrix(np.vstack(Y[:,feat_index]));
                    feature_names=feature_names[feat_index]
                    all_feature_names = np.hstack((all_feature_names, feature_names))

                    if all_X != []:
                        b = hstack((all_X, X))
                        all_X = csr_matrix(b)
                        all_Y = csr_matrix(hstack((all_Y, Y)))
                    else:
                        all_X = X
                        all_Y = Y


                elif key.find('#')!=-1:
                    L=key.split('#')
                    if L[0]=='word2vecmodel':
                        X, Y, feature_names = self.get_word2vec(tweets, tweets_train, L[2], L[3],
                                                                L[4], L[5], L[1],Google_w2v,twitter_w2v)

                    all_feature_names = np.concatenate((all_feature_names, feature_names))
                    if all_X != []:
                        b = hstack((all_X, X))
                        all_X = csr_matrix(b)
                        all_Y = csr_matrix(hstack((all_Y, Y)))
                    else:
                        all_X = X
                        all_Y = Y


                else:
                    try:
                        X,Y,feature_names = globalfeatures(key)
                    except:
                        g = globalfeatures(key)
                        X=g[0][0];Y=g[0][1];feature_names=g[0][2]
                    #print("feature length ",len(feature_names ))
                    assert (X.shape[1] == Y.shape[1]), "Length different WARNING"


                    if key!="word2vecmodel":
                        assert (X.shape[1] == len(feature_names)), "Length different WARNING"
                    all_feature_names = np.concatenate((all_feature_names, feature_names))
                    if all_X != []:


                        b=hstack((all_X,X))

                        all_X=csr_matrix(b)
                        ##attention to brakets ..
                        # https://stackoverflow.com/questions/39388902/python-concatenating-scipy-sparse-matrix
                        all_Y = csr_matrix(hstack((all_Y, Y)))
                    else:
                        all_X = X
                        all_Y = Y
            return all_X, all_Y, all_feature_names
        else:
            print ("FIRST NONE")

def make_feature_manager(TARGET,OP):

    features_manager = Features_manager(TARGET,OP)

    return features_manager
