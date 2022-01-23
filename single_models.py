import sqlite3 as lite
import os.path
import numpy as np
import re
#import nltk
import sklearn
from sklearn.externals import joblib
import pickle
from sklearn.metrics.classification import precision_recall_fscore_support, accuracy_score,confusion_matrix
from Tweet import make_tweet
#import gensim
import pandas as pd
from time import time

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", category=RuntimeWarning)


global TARGET
global TARGET_TEST

print("Running single models for a dataset of your selection...")
d=input('Enter 1 to select Hillary dataset, 2 to select Feminist dataset, 3 to select Abortion, 4 to select Atheism, 5 to select Climate dataset:  ')
ok = 0
while ok == 0:
    if d=='1' or d=='2' or d=='3' or d=='4' or d=='5':
        ok=1
        if d=='1':
            TARGET = "Hillary Clinton"
            print("Hillary Clinton dataset is selected...")
        elif d=='2':
            TARGET = "Feminist Movement"
            print("Feminist Movement dataset is selected...")
        elif d=='3':
            TARGET = "Legalization of Abortion"
            print("Legalization of Abortion dataset is selected...")
        elif d=='4':
            TARGET = "Atheism"
            print("Atheism dataset is selected...")
        elif d=='5':
            TARGET = "Climate Change is a Real Concern"
            print("Climate Change is a Real Concern dataset is selected...")
    else:
        ok = 0
        d = input("Please enter a valid number to select the dataset. Enter only 1 or 2 or 3 or 4 or 5 : ")


rf_weight=None
CV = 5
Pval_Threshold=0.05
stratified_repeated =False
str_repeats=5
R_Seed=99
modify_stance_thresh =True
stance_thresh1=0.7
stance_thresh2=0.8
n_iter_search = 15
N = 50
kval = N
tune_clfs=True
clfs=[['svm'],['rf']]

#clf_to_use=['svm']#['rf']

remove_sent_op_labels=False

TARGET_TEST =TARGET
if TARGET=="Hillary Clinton": xl="hc_"
elif TARGET=="Feminist Movement": xl="fem_"
elif TARGET=="Climate Change is a Real Concern":xl="clm_"
elif TARGET=="Legalization of Abortion":xl="ab_"
elif TARGET=="Atheism":xl="ath_"

xls_op_rf=xl+"rf_pval_"+str(Pval_Threshold)+"_cv"+str(CV)+".xlsx"
columns_op_rf = ["Op model","CV","Std","Parameters","Estimators","Min Samples Leaf","Max Features","Max Depth","Test F-score","Confusion", "Precision", "Recall", "Accuracy", "classes F-scores", "CV Time","Pvalue Features"]

xls_op_svm=xl+"svm_pval_"+str(Pval_Threshold)+"_cv"+str(CV)+".xlsx"
columns_op_svm = ["Op model","CV","Std","Parameters","C","Test F-score","Confusion", "Precision", "Recall", "Accuracy", "classes F-scores", "CV Time","Pvalue Features"]


if TARGET == "Atheism":
    oppathfile = r"atheism\\"
elif TARGET == "Feminist Movement":
    oppathfile = r"feminist\\"
elif TARGET == "Legalization of Abortion":
    oppathfile = r"abortion\\"
elif TARGET == "Climate Change is a Real Concern":
    oppathfile = r"climate\\"
elif TARGET == "Hillary Clinton":
    oppathfile = r"hc\\"

saving_cv_details=True
saving_anova_features_excel=False
saving_all_features_excel=False



analyze_scores=False
pvalue_check=True
METHOD = "allfeatures"

for clf_to_use in clfs:
    if 'rf' in clf_to_use:
        print("Running RF classifiers for the single models.............")
    elif 'svm' in clf_to_use:
        print("Running SVM classifiers on the single models.............")

    for op in range(1,13):
        if 'rf' in clf_to_use:
            print("  Tuning and Evaluating RF classifier for Model ",op,"...................")
        elif 'svm' in clf_to_use:
            print("  Tuning and Evaluating SVM classifier for Model ",op,"..................")
        #print("Model ", op)
        try:
            dframe_op_svm = pd.read_excel(xls_op_svm)
        except:
            dframe_op_svm = pd.DataFrame(columns=columns_op_svm)
        try:
            dframe_op_rf = pd.read_excel(xls_op_rf)
        except:
            dframe_op_rf = pd.DataFrame(columns=columns_op_rf)

        import Features_manager_modified

        if op in range(1,7):#
            insert_w2v = True
            Google_w2v = False
            twitter_w2v = True
            print("     Twitter word2vec is used")
        else:
            insert_w2v = True
            Google_w2v = True
            twitter_w2v = False
            print("     Google word2vec is used")

        if op in [1,2,3,7,8,9] :#
            mydic=True
            print("     Dictionary is included in pre-processing")
        else:
            mydic = False
            print("     Dictionary is NOT included in pre-processing")

        if op in [1,4,7,10]:
            insert_specific_features = True
            if modify_stance_thresh==True:
                specific_per=stance_thresh1
            else:
                specific_per = 0.7
            print("     Stance-indicative features are included with threshold 0.7")

        elif op in [2,5,8,11]:
            insert_specific_features = True
            if modify_stance_thresh == True:
                specific_per= stance_thresh2
            else:
                specific_per = 0.8
            print("     Stance-indicative features are included with threshold 0.8")

        else:
            insert_specific_features = False
            specific_per = "none"
            print("     Stance-indicative features are NOT included")

        MINDF=2




        use_saved_features=True
        remove_useless_features=False#True
        remove_repeated_features =True
        know_repeated_col=True
        remove_zero_features=True
        normalize_features=True
        use_textfile=True


        use_randomized=True

        use_cv=True#

        script_dir = os.path.dirname(__file__)
        if TARGET=="Hillary Clinton":
            rel_path = r"hc/"+"op"+str(op)+"/"
        elif TARGET=="Atheism":
            rel_path=r"atheism/"+"op"+str(op)+"/"
        elif TARGET =="Feminist Movement":
            rel_path = r"feminist/"+"op"+str(op)+"/"
        elif TARGET=="Legalization of Abortion":
            rel_path=r'abortion/'+"op"+str(op)+"/"
        elif TARGET=="Climate Change is a Real Concern":
            rel_path=r'climate/'+"op"+str(op)+"/"

        abs_file_path = os.path.join(script_dir, rel_path)
        pathfile = abs_file_path



        if TARGET=="Hillary Clinton":
            if mydic:
                rel_path = r"hc/newdic/"
            else:
                rel_path = r"hc/nodic/"
        elif TARGET=="Atheism":
            if mydic:
                rel_path=r"atheism/newdic/"
            else: rel_path=r"atheism/nodic/"
        elif TARGET =="Feminist Movement":
            if mydic: rel_path = r"feminist/newdic/"
            else: rel_path=r"feminist/nodic/"
        elif TARGET=="Legalization of Abortion":
            if mydic:
                rel_path=r'abortion/newdic/'
            else:
                rel_path=r"abortion/nodic/"

        elif TARGET=="Climate Change is a Real Concern":
            if mydic:
                rel_path=r'climate/newdic/'
            else:
                rel_path=r"climate/nodic/"

        abs_file_path = os.path.join(script_dir, rel_path)
        datapathfile = abs_file_path


        if pvalue_check == True:
            if 'svm' in clf_to_use:
                mydf = pathfile + 'Param_Pval_svm.xlsx'
            elif 'rf' in clf_to_use:
                mydf = pathfile + 'Param_Pval_rf.xlsx'
        elif pvalue_check == False:
            if 'svm' in clf_to_use:
                mydf = pathfile + 'Param_all_svm.xlsx'
            elif 'rf' in clf_to_use:
                mydf = pathfile + 'Param_all_rf.xlsx'



        #def estimator_file_generate(CLF,topn,skip_cbow,w2v_features,w2v_ctxt,w2v_min_wc,down_sampling,LDA_topics,LDA_passes):
        def estimator_file_generate(CLF, skip_cbow, w2v_features, w2v_ctxt, w2v_min_wc, down_sampling):

            if METHOD=="allfeatures":



                estimator_file = pathfile + CLF + '_best_clf' +'.pkl'
                test_score_file = pathfile + CLF + '_test_favg' +  '.pkl'
                cv_score_file = pathfile + CLF + '_cv_favg' + '.pkl'



            return estimator_file, test_score_file, cv_score_file

        #def check_clf_file(clf_to_use,topn,sp_cb,w2v_feat,w2v_ct,w2v_wc,down_sampl,LDA_topics,LDA_passes):
        def check_clf_file(clf_to_use, sp_cb, w2v_feat, w2v_ct, w2v_wc, down_sampl):


            if "rf" in clf_to_use or "all" in clf_to_use:
                #estimator_file, test_score_file, cv_score_file = estimator_file_generate("rf",topn,sp_cb,w2v_feat,w2v_ct,w2v_wc,down_sampl,LDA_topics,LDA_passes)
                estimator_file, test_score_file, cv_score_file = estimator_file_generate("rf",sp_cb,w2v_feat,w2v_ct,w2v_wc,down_sampl)

                if os.path.isfile(test_score_file):
                    favg_rf = joblib.load(test_score_file)
                    if os.path.isfile(estimator_file):
                        tuned_rf = joblib.load(estimator_file)
                        if os.path.isfile(cv_score_file):
                            cv_rf = joblib.load(cv_score_file)
                            #print('favg_rf', favg_rf)
                            #print('cv_rf', cv_rf)
                            #print('tuned_rf', tuned_rf)
                            return favg_rf, cv_rf, tuned_rf

            if "svm" in clf_to_use or "all" in clf_to_use:
                #estimator_file, test_score_file, cv_score_file = estimator_file_generate("svm",topn,sp_cb,w2v_feat,w2v_ct,w2v_wc,down_sampl,LDA_topics,LDA_passes)
                estimator_file, test_score_file, cv_score_file = estimator_file_generate("svm",sp_cb,w2v_feat,w2v_ct,w2v_wc,down_sampl)

                if os.path.isfile(test_score_file):
                    favg_svm = joblib.load(test_score_file)
                    if os.path.isfile(estimator_file):
                        tuned_svm = joblib.load(estimator_file)
                        if os.path.isfile(cv_score_file):
                            cv_svm = joblib.load(cv_score_file)
                            #print('favg_svm', favg_svm)
                            #print('cv_svm', cv_svm)
                            #print('tuned_svm', tuned_svm)
                            return favg_svm, cv_svm, tuned_svm

            return None , None , None

        def read_tweets(data,filter=None):


            if os.path.isfile(datapathfile+data+filter+'.pkl') :
                try:
                    tweets= joblib.load(datapathfile+data+filter+'.pkl')
                except:
                    tweets = joblib.load(datapathfile + data + filter + '.pkl',allow_pickle=True)
                try:
                    favor = joblib.load(datapathfile+data +"_favor"+ filter + '.pkl')
                except:
                    favor = joblib.load(datapathfile + data + "_favor" + filter + '.pkl',allow_pickle=True)
                try:
                    against = joblib.load(datapathfile+data +"_against"+ filter + '.pkl')
                except:
                    against = joblib.load(datapathfile+data +"_against"+ filter + '.pkl',allow_pickle=True)
                try:
                    none = joblib.load(datapathfile + data + "_none" + filter + '.pkl')
                except:
                    none = joblib.load(datapathfile+data +"_none"+ filter + '.pkl',allow_pickle=True)

                return tweets,against,favor,none

            if os.path.isfile(datapathfile+data + filter + '.npy'):
                try:
                    tweets = np.load(datapathfile+data + filter + '.npy')
                except:
                    tweets = np.load(datapathfile + data + filter + '.npy',allow_pickle=True)
                try:
                    favor = np.load(datapathfile + data + "_favor" + filter + '.npy')
                except:
                    favor = np.load(datapathfile+data + "_favor" + filter + '.npy',allow_pickle=True)
                try:
                    against = np.load(datapathfile + data + "_against" + filter + '.npy')
                except:
                    against = np.load(datapathfile+data + "_against" + filter + '.npy',allow_pickle=True)
                try:
                    none = np.load(datapathfile + data + "_none" + filter + '.npy')
                except:
                    none = np.load(datapathfile+data + "_none" + filter + '.npy',allow_pickle=True)

                return tweets, against, favor, none

            db=lite.connect("Stance.db")
            cur=db.cursor()
            db.commit()
            tweets=[]
            against=[]
            none=[]
            favor=[]

            if filter != "none":
                if data=="tweets_train":
                    cur.execute("SELECT ID, Target, Tweet, Stance,Opinion,Sentiment FROM tweets_train where Target=? order by ID",[filter])
                else:
                    cur.execute("SELECT ID,Target, Tweet, Stance,Opinion,Sentiment FROM tweets_test where Target=? order by ID",[filter])

            else:
                if data=="tweets_train":
                    cur.execute("SELECT ID, Target,Tweet,Stance,Opinion,Sentiment FROM tweets_train order by ID")
                else:
                    cur.execute("SELECT ID,Target,Tweet,Stance,Opinion,Sentiment FROM tweets_test order by ID")


            for tweet in cur.fetchall():
                sentiment = return_sentiment(tweet[5])
                opiniontowards= return_opinion(tweet[4])
                stance=return_stance(tweet[3])
                this_tweet=make_tweet(mydic,tweet[0], tweet[2],tweet[1], opiniontowards,sentiment, stance)

                tweets.append(this_tweet)
                if stance==2:
                    against.append(this_tweet)
                elif stance==1:
                    favor.append(this_tweet)
                elif stance==0:
                    none.append(this_tweet)
                else:
                    print("STANCE NOT KNOWN")

            tweets = np.array(tweets)
            none=np.array(none)
            favor=np.array(favor)
            against=np.array(against)

            if data=="tweets_train":
                np.save(datapathfile+'tweets_train'+filter+'.npy',tweets)
                np.save(datapathfile+ 'tweets_train_favor' + filter+'.npy',favor)
                np.save( datapathfile+'tweets_train_against' + filter +'.npy',against)
                np.save( datapathfile+'tweets_train_none' + filter +'.npy',none)
            else:
                np.save( datapathfile+'tweets_test'+filter+'.npy',tweets)
                np.save( datapathfile+'tweets_test_favor'+filter+'.npy',favor)
                np.save( datapathfile+'tweets_test_against'+filter+'.npy',against)
                np.save( datapathfile+'tweets_test_none'+filter+'.npy',none)
            return tweets,against,favor,none


        def return_sentiment(sent):
            label_sent=0
            if sent=="neg":
                label_sent=-1
            elif sent=="pos":
                label_sent=1
            elif sent=="other":
                label_sent=0
            else :label_sent=2
            return label_sent


        def return_opinion(opin):
            label_opin=0
            if opin=="1.  The tweet explicitly expresses opinion about the target, a part of the target, or an aspect of the target.":
                label_opin=1
            elif opin== "2. The tweet does NOT expresses opinion about the target but it HAS opinion about something or someone other than the target.":
                label_opin=2
            elif opin== "3.  The tweet is not explicitly expressing opinion. (For example, the tweet is simply giving information.)":
                label_opin=3

            return label_opin

        def return_stance(stance):
            label_stance=0
            if stance=="AGAINST":
                label_stance=2
            elif stance=="FAVOR":
                label_stance=1
            elif stance=="NONE":
                label_stance=0

            return label_stance


        tweets_train,tweets_train_against,tweets_train_favor,tweets_train_none=read_tweets("tweets_train",filter=TARGET)
        tweets_test,tweets_test_against,tweets_test_favor,tweets_test_none=read_tweets("tweets_test",filter=TARGET_TEST)


        from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer

        def hashtag_analysis(tweets_favor, tweets_against, tweets_none):
            favor=[]
            for t in tweets_favor:
                fav=""
                favor_elements=re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
                for elem in favor_elements:
                    elem=elem.lstrip('<hashtag>');elem=elem.rstrip('</hashtag>');elem=elem.strip()
                    fav+=elem+" "
                favor.append(fav)

            against=[]
            for t in tweets_against:
                ag=""
                against_elements = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
                for elem in against_elements:
                    elem = elem.lstrip('<hashtag>');elem = elem.rstrip('</hashtag>');elem = elem.strip()
                    ag+=elem+" "
                against.append(ag)
            none = []
            for t in tweets_none:
                nn = ""
                none_elements = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
                for elem in none_elements:
                    elem = elem.lstrip('<hashtag>');elem = elem.rstrip('</hashtag>');elem = elem.strip()
                    nn += elem + " "
                none.append(nn)

            countword = CountVectorizer(ngram_range=(1, 1), stop_words='english', max_features=50)
            favor_count = countword.fit_transform(favor)
            features = countword.get_feature_names()
            against_count = countword.transform(against)
            none_count = countword.transform(none)
            favor_feat = set([])
            against_feat = set([])
            for i in range(favor_count.shape[1]):
                if ((np.sum(favor_count.todense(), axis=0)[:, i] - np.sum(against_count.todense(), axis=0)[:, i]) >= 2):
                    favor_feat.add(features[i])
                if ((np.sum(against_count.todense(), axis=0)[:, i] - np.sum(favor_count.todense(), axis=0)[:, i]) >= 4):
                    against_feat.add(features[i])

            against_count = countword.fit_transform(against)
            features = countword.get_feature_names()
            favor_count = countword.transform(favor)
            none_count = countword.transform(none)
            for i in range(against_count.shape[1]):
                if ((np.sum(favor_count.todense(), axis=0)[:, i] - np.sum(against_count.todense(), axis=0)[:, i]) >= 2):
                    favor_feat.add(features[i])
                if ((np.sum(against_count.todense(), axis=0)[:, i] - np.sum(favor_count.todense(), axis=0)[:, i]) >= 4):
                    against_feat.add(features[i])
            favor_feat = list(favor_feat)
            against_feat = list(against_feat)
            return favor_count, against_count, none_count, features, favor_feat, against_feat

        favortags_file=pathfile+"favortags_file.npy"
        againsttags_file=pathfile+"againsttags_file.npy"

        favor_tags=None
        against_tags=None

        if os.path.isfile(favortags_file):
            favor_tags=np.load(favortags_file)
            if os.path.isfile(againsttags_file):
                against_tags = np.load(againsttags_file)
        else :
            _,_,_,_,favor_tags,against_tags=hashtag_analysis(tweets_train_favor,tweets_train_against,tweets_train_none)

        def specific_words(train_favor,train_against,train_none):
            per_ag = per_fav =per_target = specific_per
            include_once_items=False
            hash="not effective"
            fav_tags=[];ag_tags=[];none_tags=[];all_tags=[];fav_notags=[];ag_notags=[];none_notags=[];all_notags=[]
            if hash==True or hash=="not effective":
                for t in train_favor:
                    favor_elements=re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
                    if favor_elements:
                        for elem in favor_elements:
                            elem=elem.lstrip('<hashtag>');elem=elem.rstrip('</hashtag>');elem=elem.strip()
                            fav_tags.append(elem.lower());all_tags.append(elem.lower())
                for t in train_against:
                    against_elements = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
                    if against_elements:
                        for elem in against_elements:
                            elem = elem.lstrip('<hashtag>');elem = elem.rstrip('</hashtag>');elem = elem.strip()
                            ag_tags.append(elem.lower());all_tags.append(elem.lower())

                for t in train_none:
                    none_elements = re.findall(r'<hashtag>[\s+\w+\._?!]+</hashtag>', t.text_hash)
                    if none_elements:
                        for elem in none_elements:
                            elem = elem.lstrip('<hashtag>');elem = elem.rstrip('</hashtag>');elem = elem.strip()
                            none_tags.append(elem.lower());all_tags.append(elem.lower())
            if hash==False or hash=="not effective":
                for t in train_favor:
                    t=re.sub(r'<hashtag>[\s+\w+\._?!]+</hashtag>', '', t.text_hash)
                    t=re.sub(r'<user>[\s+\w+\._?!]+</user>', '', t)
                    t=t.split()
                    for e in t:
                        if e.isalpha():
                            fav_notags.append(e.lower());all_notags.append(e.lower())
                for t in train_against:
                    t=re.sub(r'<hashtag>[\s+\w+\._?!]+</hashtag>', '', t.text_hash)
                    t=re.sub(r'<user>[\s+\w+\._?!]+</user>', '', t)
                    t=t.split()
                    for e in t:
                        if e.isalpha():
                            ag_notags.append(e.lower());all_notags.append(e.lower())
                for t in train_none:
                    t=re.sub(r'<hashtag>[\s+\w+\._?!]+</hashtag>', '', t.text_hash)
                    t=re.sub(r'<user>[\s+\w+\._?!]+</user>', '', t)
                    t=t.split()
                    for e in t:
                        if e.isalpha():
                            none_notags.append(e.lower());all_notags.append(e.lower())
            ag = ag_tags + ag_notags
            fav = fav_tags + fav_notags
            none =  none_tags + none_notags
            all = all_tags + all_notags
            ag = [x for x in ag if x]
            fav = [x for x in fav if x]
            none = [x for x in none if x]
            all = [x for x in all if x]
            from collections import Counter
            count_ags = Counter(ag)
            count_favs = Counter(fav)
            count_nones = Counter(none)
            count_alls = Counter(all)
            count_ag= Counter()
            for k, v in count_ags.items():
                if v > 1:
                    count_ag[k] = v
            count_fav = Counter()
            for k, v in count_favs.items():
                if v > 1:
                    count_fav[k] = v
            count_none = Counter()
            for k, v in count_nones.items():
                if v > 1:
                    count_none[k] = v
            count_all = Counter()
            for k, v in count_alls.items():
                if v > 1:
                    count_all[k] = v


            ag_list = [];fav_list = [];target_list = []
            import math
            if include_once_items:
                c = count_alls
            else:
                c = count_all
            for k, v in c.items():
                thresh_ag = int(math.ceil(per_ag * v))
                thresh_fav = int(math.ceil(per_fav * v))
                thresh_target = int(math.ceil(per_target * v))
                try:
                    if count_ags[k] >= thresh_ag:
                        ag_list.append(k)
                except:
                    pass
                try:
                    if count_favs[k] >= thresh_fav:
                        fav_list.append(k)
                except:
                    pass
                try:
                    if (count_ags[k] + count_favs[k]) >= thresh_target:
                        target_list.append(k)
                except:
                    pass

            return ag_list,fav_list,target_list

        if insert_specific_features == True:
            against_list_file=pathfile+"againstlist.npy"
            favor_list_file=pathfile+"favorlist.npy"
            target_list_file=pathfile+"targetlist.npy"
            if os.path.isfile( target_list_file):
                target_list=np.load( target_list_file)
                if os.path.isfile(favor_list_file):
                    favor_list = np.load(favor_list_file)
                    if os.path.isfile(against_list_file):
                        against_list = np.load(str(against_list_file))
            else:
                against_list, favor_list, target_list = specific_words(tweets_train_favor, tweets_train_against,
                                                                       tweets_train_none)

        else:
            against_list=None; favor_list=None; target_list=None


        def DSD(tweets_favor, tweets_against, tweets_none):
            tweet_favor = []
            tweet_against = []
            tweet_none = []
            for tweet in tweets_favor:
                tweet_favor.append(tweet.text_nopunc)
            for tweet in tweets_against:
                tweet_against.append(tweet.text_nopunc)
            for tweet in tweets_none:
                tweet_none.append(tweet.text_nopunc)

            countword = CountVectorizer(ngram_range=(1, 1), stop_words='english', max_features=50)
            favor_count = countword.fit_transform(tweet_favor)
            features = countword.get_feature_names()
            against_count = countword.transform(tweet_against)
            none_count = countword.transform(tweet_none)
            favor_feat = set([])
            against_feat = set([])
            for i in range(favor_count.shape[1]):
                if ((np.sum(favor_count.todense(), axis=0)[:, i] - np.sum(against_count.todense(), axis=0)[:, i]) >= 3):
                    favor_feat.add(features[i])
                if ((np.sum(against_count.todense(), axis=0)[:, i] - np.sum(favor_count.todense(), axis=0)[:, i]) >= 10):
                    against_feat.add(features[i])

            against_count = countword.fit_transform(tweet_against)
            features = countword.get_feature_names()
            favor_count = countword.transform(tweet_favor)
            none_count = countword.transform(tweet_none)
            for i in range(against_count.shape[1]):
                if ((np.sum(favor_count.todense(), axis=0)[:, i] - np.sum(against_count.todense(), axis=0)[:, i]) >= 3):
                    favor_feat.add(features[i])
                if ((np.sum(against_count.todense(), axis=0)[:, i] - np.sum(favor_count.todense(), axis=0)[:, i]) >= 10):
                    against_feat.add(features[i])
            favor_feat = list(favor_feat)
            against_feat = list(against_feat)
            return favor_count, against_count, none_count, features, favor_feat, against_feat

        favorwords_file=pathfile+"favorwords_file.npy"
        againstwords_file = pathfile+"againstwords_file.npy"
        against_words=None;favor_words=None
        if os.path.isfile(favorwords_file):
            favor_words=np.load(favorwords_file)
            if os.path.isfile(againstwords_file):
                against_words = np.load(againstwords_file)

        else:
            _,_,_,_,favor_words,against_words=DSD(tweets_train_favor,tweets_train_against,tweets_train_none)



        if remove_sent_op_labels == True:
            ft1 = ["hashtag", "screen", "num_hashtag", "num_screen", "posfeat", "posfeat1"]
        else:
            ft1=["sentiment_labels","opinion_labels","hashtag","screen","num_hashtag","num_screen","posfeat","posfeat1"]
        ft2=["word2vecmodel"]
        ft3=['ngrams',"chars"]
        ft4=["sentdal","sentafinn",
                     "gi",
                     "hashtag-idf",'ngrams-idf',\
                     "posfeat2","tweet_len"]
        ft5=["avg_word_len","seq_vowel","capital_words","num_punc","find_neg","stan_parser",\
                     "browncluster","mpqa","bing_lius","NRC_hashtag","NRC_sent140"]
        ft6=["twise-pos","twise-hashaffneg","twise-hashaffneg-bi","twise-sent140affneg","twise-sent140affneg-bi","twise-mpqa",\
                     "twise-bingliu","twise-sentiwordnet"]
        ft7=["elongated","allcaps"]
        ft8=["ngrams-bin","hashtag-bin","screen-bin"]
        ft9=[ "bowtags","bowmentions","BoWinHashtag","targetandnoinhashtag",\
                     "target-bin","targetintweet"]

        ft11=["mpqa_new","wordnet_new","sentiwn_new","polar_mpqa_new","ngrams_target_new","cgrams_target_new",
                      "target_detection_new"]

        allfeatures = ft1 + ft3+ ft4 + ft5 + ft6 + ft7 + ft8 + ft11
        if insert_w2v==True :
            allfeatures+=ft2
        if insert_specific_features == True:
            allfeatures += ft9



        feature_manager=Features_manager_modified.make_feature_manager(TARGET,op)
        stance_train=feature_manager.get_stance(tweets_train)
        stance_test=feature_manager.get_stance(tweets_test)


        #def computescores(stance_train,stance_test,allword2vec,allLDA,LDA_topics,LDA_passes,w2v_features,w2v_min_wc,w2v_ctxt,down_sampling,skip_cbow,tweets_test,tweets_train,target,featureset,favor_words, against_words,favor_tags,against_tags,clf_to_use,N=None,method=None):
        def computescores(stance_train, stance_test, allword2vec, w2v_features,
                              w2v_min_wc, w2v_ctxt, down_sampling, skip_cbow, tweets_test, tweets_train, target,
                              featureset, favor_words, against_words, favor_tags, against_tags, clf_to_use, N=None,
                              method=None):



            trainfile = pathfile + "extracted_features_train_matrix" + '.pkl'
            testfile = pathfile + "extracted_features_test_matrix"+ '.pkl'
            featfile = pathfile + "extracted_features_names"+'.pkl'

            def removing_repeated_features(x_train,x_test,feat):
                def method_1(x_train,x_test,feat):
                    X=np.vstack((x_train.toarray(),x_test.toarray()))
                    X=X.T
                    x=x_train.shape[0]
                    X = np.ascontiguousarray(X)
                    from time import time
                    start = time()
                    unique_x, ind = np.unique(X.view([('', X.dtype)] * X.shape[0]), return_index=True)
                    from time import time
                    start = time()
                    feat=[feat[j] for j in list(ind)]
                    for e in range(X.shape[0]):
                        if e == 0:
                            Y = unique_x['f' + str(e)]
                        else:
                            Y = np.vstack((Y, unique_x['f' + str(e)]))
                    x_train=Y[:x,:]
                    x_test=Y[x:,:]
                    return x_train, x_test, feat

                def method_2(x_train,x_test,feat):
                    try:
                        X=np.vstack((x_train.toarray(),x_test.toarray()))
                    except:
                        X=np.vstack((x_train,x_test))

                    all_ind_set=set(np.array(range(x_train.shape[1])))
                    unique_x, ind , coun = np.unique(X, axis=1, return_index=True,return_counts=True)
                    ind_set=set(ind)
                    not_unique=all_ind_set-ind_set
                    removed_feat = [feat[j] for j in list(not_unique)]
                    feat=[feat[j] for j in list(ind)]
                    x = x_train.shape[0]
                    x_train = unique_x[:x, :]
                    x_test = unique_x[x:, :]
                    return x_train,x_test,feat
                x_train, x_test, feat=method_2(x_train, x_test, feat)
                return x_train,x_test , feat
            def removing_zero_features(x_train,x_test,feat):
                try:
                    X = np.vstack((x_train.toarray(), x_test.toarray()))
                except:
                    X = np.vstack((x_train, x_test))

                ind=[]
                for i in range(X.shape[1]):
                    if X[:,i].any():
                        ind.append(i)
                all_ind_set = set(np.array(range(x_train.shape[1])))
                ind_set = set(ind)
                not_unique = all_ind_set - ind_set
                removed_feat = [feat[j] for j in list(not_unique)]
                feat = [feat[j] for j in list(ind)]
                x = x_train.shape[0]
                ind=np.array(ind)
                X=X[:,ind]
                x_train = X[:x, :]
                x_test = X[x:, :]
                if saving_all_features_excel:
                    filename = "All_Features_"+xl+"_op_"+str(op)+".xlsx"
                    columns = ['All Features']
                    df = pd.DataFrame(feat, columns=columns)
                    df.to_excel(filename, index=False)
                return x_train, x_test, feat

            def knowing_repeated_col(x_train,x_test,feat):
                try:
                    X = np.vstack((x_train.toarray(), x_test.toarray()))
                except:
                    X = np.vstack((x_train, x_test))
                def method_1():
                    m=[]
                    for i in range(X.shape[1]):
                        l = []
                        for j in range(i + 1, X.shape[1]):
                            if np.array_equal(X[:, i], X[:, j]):
                                l.append(j)
                        if l != []: m.append(set(l + [i]))
                    k=[]
                    for e in m:
                        for ee in m:
                            if e.intersection(ee) != set([]):
                                e = e.union(ee)
                        if list(e) not in k: k.append(list(e))
                    feat_repeated=[]
                    for e in k:
                        y = [feat[ee] for ee in e]
                        feat_repeated.append(y)
                def method_2(X,feat,x_train,x_test):
                    def unique_columns(data):
                        dt = np.dtype((np.void, data.dtype.itemsize * data.shape[0]))
                        dataf = np.asfortranarray(data).view(dt)
                        u, uind = np.unique(dataf, return_inverse=True)
                        u = u.view(data.dtype).reshape(-1, data.shape[0]).T
                        return (u, uind)

                    unique, uind = unique_columns(X)
                    l = [np.where(uind == xx)[0] for xx in range(unique.shape[0])]
                    feat_repeated = []
                    for e in l:
                        y = [feat[ee] for ee in e]
                        if len(y) != 1: feat_repeated.append(y)
                    x=x_train.shape[0]
                    x_train=unique[:x,:]
                    x_test = unique[x:, :]

                    return x_train,x_test,feat_repeated
                x_train,x_test,feat_repeated=method_2(X,feat,x_train,x_test)


            def removing_useless_features(x_train, x_test, feat):
                try:
                    x_train = x_train.toarray()
                except:
                    pass
                means = np.mean(x_train, axis=0)
                stds = np.std(x_train, axis=0)
                useless = np.where(stds == 0)[0]
                x_train = np.delete(x_train, useless, 1)
                feat = np.delete(feat, useless, 0)

                try:
                    x_test = np.delete(x_test.toarray(), useless, 1)
                except:
                    x_test = np.delete(x_test, useless, 1)
                return x_train, x_test, feat

            def normalizing_features(x_train, x_test,feat):

                try:
                    x_train = x_train.astype(
                        'float64')
                except:
                    x_train = x_train.toarray().astype('float64')
                try:
                    x_test = x_test.astype(
                        'float64')
                except:
                    x_test = x_test.toarray().astype('float64')

                def separate_mean_std(x_train,x_test):
                    means_train = np.mean(x_train, axis=0)
                    stds_train = np.std(x_train, axis=0)
                    means_test = np.mean(x_test, axis=0)
                    stds_test = np.std(x_test, axis=0)
                    for e in range(x_train.shape[1]):
                        if stds_test[e] != 0 and stds_train[e]!=0:
                            x_test[:, e] = (x_test[:, e] - means_test[e]) / stds_test[e]
                            x_train[:, e] = (x_train[:, e] - means_train[e]) / stds_train[e]
                        else:
                            x_test[:, e] = x_test[:, e] - means_test[e]
                            x_train[:, e] = x_train[:, e] - means_train[e]
                    return x_train,x_test

                def one_mean_for_traintest(x_train,x_test):
                    X=np.vstack((x_train,x_test))
                    means = np.mean(X,axis=0)
                    stds = np.std(X, axis=0)

                    for e in range(x_train.shape[1]):
                        if stds[e] != 0 :
                            x_test[:, e] = (x_test[:, e] - means[e]) / stds[e]
                            x_train[:, e] = (x_train[:, e] - means[e]) / stds[e]
                        else:
                            x_test[:, e] = x_test[:, e] - means[e]
                            x_train[:, e] = x_train[:, e] - means[e]
                    return x_train,x_test

                x_train,x_test=one_mean_for_traintest(x_train, x_test)

                from scipy.sparse import csr_matrix

                x_train = csr_matrix(x_train)
                x_test = csr_matrix(x_test)

                return x_train, x_test

            if use_saved_features==True:
                if os.path.isfile(trainfile):
                    x_train = joblib.load(trainfile)
                    x_test = joblib.load(testfile)
                    feat = joblib.load(featfile)
                    try:
                        x_train=x_train.toarray()
                        x_test=x_test.toarray()
                    except:
                        pass

                else:
                    x_train, x_test, feat = feature_manager.create_feature_space(MINDF, against_list, favor_list,
                                                                                 target_list, Google_w2v, twitter_w2v,
                                                                                 generate_allfeat, w2v_features,
                                                                                 w2v_min_wc, w2v_ctxt, down_sampling,
                                                                                 skip_cbow, tweets_test, tweets_train,
                                                                                 target, featureset,
                                                                                 favor_words, against_words, favor_tags,
                                                                                 against_tags,2)
                    # x_train, x_test, feat = feature_manager.create_feature_space(MINDF,against_list,favor_list,target_list,Google_w2v,twitter_w2v,generate_allfeat,w2v_features,w2v_min_wc,w2v_ctxt,down_sampling,skip_cbow,tweets_test, tweets_train, target, featureset,
                    #                                                              favor_words, against_words, favor_tags,
                    #                                                              against_tags, topn, n_topics, n_passes, 2)
                    TOTAL_NUM_FEAT = len(feat)
                    if remove_repeated_features == True:
                        x_train, x_test, feat=removing_repeated_features(x_train,x_test,feat)
                    if remove_zero_features==True:
                        x_train, x_test, feat =removing_zero_features(x_train, x_test, feat)
                    if know_repeated_col==True:
                        knowing_repeated_col(x_train,x_test,feat)

                    if pvalue_check==True:
                        from sklearn import feature_selection
                        from time import time
                        start = time()
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fs = sklearn.feature_selection.SelectKBest(feature_selection.f_classif, k=x_train.shape[1])

                        fs.fit(x_train, stance_train)
                        Pvalues = fs.pvalues_
                        pvalue_time=time()-start
                        #print("Pvalue test took %.2f seconds " % (pvalue_time))
                        sig = []
                        i = 0
                        for p in Pvalues:
                            if p < Pval_Threshold:
                                sig.append(i)
                            i = i + 1
                        feat = [feat[j] for j in sig]
                        #print("Pvalue number of features selected  ", len(feat))
                        x_train = x_train[:, sig]
                        x_test = x_test[:, sig]

                        if saving_anova_features_excel:
                            filename = "Anova_Features_" + xl+ "_op_" + str(op) + ".xlsx"
                            columns = ['Anova Features']
                            df = pd.DataFrame(feat, columns=columns)
                            df.to_excel(filename, index=False)


                    if normalize_features == True:
                        x_train, x_test=normalizing_features(x_train,x_test,feat)

                    NUM_FEAT = len(feat)
                    joblib.dump(x_train, trainfile, protocol=2)
                    joblib.dump(x_test, testfile, protocol=2)
                    joblib.dump(feat, featfile, protocol=2)

            else:
                #x_train,x_test,feat=feature_manager.create_feature_space(MINDF,against_list,favor_list,target_list,Google_w2v,twitter_w2v,allword2vec,w2v_features,w2v_min_wc,w2v_ctxt,down_sampling,skip_cbow,tweets_test,tweets_train,target,featureset,favor_words, against_words,favor_tags,against_tags,topn,n_topics,n_passes,2)
                x_train,x_test,feat=feature_manager.create_feature_space(MINDF,against_list,favor_list,target_list,Google_w2v,twitter_w2v,allword2vec,w2v_features,w2v_min_wc,w2v_ctxt,down_sampling,skip_cbow,tweets_test,tweets_train,target,featureset,favor_words, against_words,favor_tags,against_tags,2)

                if remove_useless_features == True:
                    x_train, x_test, feat = removing_useless_features(x_train, x_test, feat)
                if remove_repeated_features == True:
                    x_train, x_test, feat = removing_repeated_features(x_train, x_test, feat)

                if normalize_features == True:
                    x_train, x_test = normalizing_features(x_train, x_test,feat)
                NUM_FEAT=len(feat)
                joblib.dump(x_train,trainfile,protocol=2)
                joblib.dump(x_test,testfile,protocol=2)
                joblib.dump(feat, featfile, protocol=2)



            def clf_score(clf_model,param_grid,stance_train,stance_test,col_FEATURE=None,model_name=None):

                if model_name=="svm":
                    USE_randomized=False
                else:
                    USE_randomized=use_randomized
                if param_grid=={}:
                    USE_randomized = False


                from sklearn.metrics import f1_score
                from sklearn.metrics import make_scorer

                if model_name=="mnb":
                    X_train=np.absolute(x_train)
                    X_test=np.absolute(x_test)
                else:
                    X_train=x_train
                    X_test=x_test

                def avg_score(xx,stance_train):
                    prec, recall, f, support = precision_recall_fscore_support(xx, stance_train, labels=[2, 1, 0],
                                                                               beta=1)
                    faverage=(f[0]+f[1])*0.5
                    return faverage

                f1_scorer = make_scorer(avg_score,greater_is_better=True)

                try:
                    if use_cv == False:
                        start = time()
                        clf_model.fit(X_train, stance_train)
                        diff_time = time() - start
                        print("Fitting model took %.2f seconds " % (diff_time))
                    else:

                        from time import time
                        if USE_randomized==True:
                            from sklearn.model_selection import StratifiedKFold
                            if stratified_repeated==False:
                                clf_model=sklearn.model_selection.RandomizedSearchCV(verbose=0,estimator=clf_model, param_distributions=param_grid, cv=StratifiedKFold(n_splits=CV, random_state=R_Seed, shuffle=True),scoring=f1_scorer,n_iter=n_iter_search,random_state=R_Seed)
                            else:
                                from sklearn.model_selection import RepeatedStratifiedKFold
                                clf_model = sklearn.model_selection.RandomizedSearchCV(verbose=0, estimator=clf_model,param_distributions=param_grid,
                                                                                       cv=RepeatedStratifiedKFold(n_splits=CV,random_state=R_Seed,
                                                                                                          n_repeats=str_repeats),scoring=f1_scorer, n_iter=n_iter_search,random_state=R_Seed)
                        elif USE_randomized==False:

                            from sklearn.model_selection import GridSearchCV
                            from sklearn.model_selection import StratifiedKFold

                            if stratified_repeated == False:
                                from time import time
                                clf_model = GridSearchCV(estimator=clf_model, param_grid=param_grid, cv=StratifiedKFold(n_splits=CV, random_state=R_Seed, shuffle=True),scoring=f1_scorer)
                            else:
                                from sklearn.model_selection import RepeatedStratifiedKFold
                                clf_model = GridSearchCV(estimator=clf_model, param_grid=param_grid,
                                                         cv=RepeatedStratifiedKFold(n_splits=CV, random_state=R_Seed, n_repeats=str_repeats),
                                                         scoring=f1_scorer)

                        from time import time
                        start = time()
                        clf_model.fit(X_train, stance_train)
                        diff_time = time() - start
                        print("     Fitting model took %.2f seconds " % (diff_time))
                        try:
                            if model_name=="rf":
                                trees=[estimator.tree_.max_depth for estimator in clf_model.best_estimator_]
                        except: pass

                except:
                    if use_cv == False:
                        start = time()
                        clf_model.fit(X_train.toarray(), stance_train)
                        diff_time = time() - start
                        print("     Fitting model took %.2f seconds " % (diff_time))

                    else:


                        from time import time
                        if USE_randomized==True:
                            from sklearn.model_selection import StratifiedKFold
                            if stratified_repeated==False:
                                clf_model=sklearn.model_selection.RandomizedSearchCV(verbose=0,estimator=clf_model, param_distributions=param_grid, cv=StratifiedKFold(n_splits=CV, random_state=R_Seed, shuffle=True),scoring=f1_scorer,n_iter=n_iter_search,random_state=R_Seed)
                            else:
                                from sklearn.model_selection import RepeatedStratifiedKFold
                                clf_model=sklearn.model_selection.RandomizedSearchCV(verbose=0,estimator=clf_model, param_distributions=param_grid, cv=RepeatedStratifiedKFold(n_splits=CV, random_state=R_Seed, n_repeats=str_repeats),scoring=f1_scorer,n_iter=n_iter_search,random_state=R_Seed)


                        elif USE_randomized==False:
                            from sklearn.model_selection import GridSearchCV
                            from sklearn.model_selection import StratifiedKFold
                            if stratified_repeated == False:
                               clf_model = GridSearchCV(estimator=clf_model, param_grid=param_grid, cv=StratifiedKFold(n_splits=CV, random_state=R_Seed, shuffle=True),scoring=f1_scorer)
                            else:
                                from sklearn.model_selection import RepeatedStratifiedKFold
                                clf_model = GridSearchCV(estimator=clf_model, param_grid=param_grid,
                                                        cv=RepeatedStratifiedKFold(n_splits=CV, random_state=R_Seed, n_repeats=str_repeats),
                                                       scoring=f1_scorer)

                        try:
                            start = time()
                            clf_model.fit(X_train, stance_train)
                            diff_time=time()-start
                            print("     Fitting model took %.2f seconds " % (diff_time))
                        except:
                            from sklearn.model_selection import GridSearchCV
                            from sklearn.model_selection import StratifiedKFold
                            if stratified_repeated == False:
                                clf_model = GridSearchCV(estimator=clf_model, param_grid=param_grid, cv=StratifiedKFold(n_splits=CV, random_state=R_Seed, shuffle=True), scoring=f1_scorer)
                            else:
                                from sklearn.model_selection import RepeatedStratifiedKFold
                                clf_model = GridSearchCV(estimator=clf_model, param_grid=param_grid, cv=RepeatedStratifiedKFold(n_splits=CV, random_state=R_Seed, n_repeats=str_repeats), scoring=f1_scorer)

                            start=time()
                            try:
                                clf_model.fit(X_train, stance_train)
                            except:
                                clf_model.fit(X_train.toarray(),stance_train)
                            diff_time = time() - start


                if saving_cv_details==True:
                    params = clf_model.cv_results_['params']
                    means = clf_model.cv_results_['mean_test_score']
                    stds = clf_model.cv_results_['std_test_score']
                    cvfile=pathfile +'cv_rf_'+str(n_iter_search)+"iter.xlsx"
                    try:
                        df = pd.read_excel(cvfile)
                    except:
                        df = pd.DataFrame(columns=['Estimators', 'Min samples leaf', 'Max features', 'Max depth', 'mean_test_score','std_test_score'])
                    for mean, std, param in zip(means, stds, params):
                        if clf_to_use==['rf']:
                            if param['max_depth']==None:
                                MDD="None"
                            else:
                                MDD=str(param['max_depth'])

                            if param['max_features']==None:
                                MFF="None"
                            else:
                                MFF=str(param['max_features'])
                            df.loc[len(df)] = [str(param['n_estimators']), str(param['min_samples_leaf']),
                                           MFF, MDD, mean, std]
                            df.to_excel(cvfile, encoding='utf-8', index=False)

                test_predict = clf_model.predict(X_test)
                try:
                    test_predict = clf_model.predict(X_test)
                except:
                    test_predict = clf_model.predict(X_test.toarray())

                prec, recall, f, support = precision_recall_fscore_support(stance_test, test_predict, labels=[2, 1, 0],
                                                                           beta=1)
                Conf=confusion_matrix(stance_test, test_predict, labels=[2, 1, 0])
                #print("Confusion matrix ",Conf)
                #print("precision ",prec)
                #print("recall ",recall)
                #print("support ",support)
                accuracy = accuracy_score(stance_test, test_predict)
                #print ("accuracy ", accuracy)
                #print ("f ", f)
                #print ("favg test set : ", (f[0] + f[1]) * 0.5)
                favg = (f[0] + f[1]) * 0.5
                c = np.where(stance_test != test_predict)[0]
                #print("There are ", len(c), " predicted wrong out of ", len(stance_test))
                if analyze_scores==True:
                    Error_file ="error_"+ xl + "_op_"+str(op)+"_"+clf_to_use[0]+".xlsx"
                    True_file = "true_"+ xl + "_op_"+str(op)+"_"+clf_to_use[0]+".xlsx"
                    columns = ["Tweet_ID", "TrueLabel", "Predicted", "Tweet"]
                    try:
                        error_frame = pd.read_excel(Error_file)
                    except:
                        error_frame = pd.DataFrame(columns=columns)
                    try:
                        true_frame = pd.read_excel(True_file)
                    except:
                        true_frame = pd.DataFrame(columns=columns)
                    wrong = []
                    for i in range(len(test_predict)):
                        if test_predict[i] != stance_test[i]:
                            wrong.append(i + 1)
                            error_frame.loc[len(error_frame)] = [i + 1, stance_test[i], test_predict[i], tweets_test[i].text_raw]
                            error_frame.to_excel(Error_file, encoding='utf-8', index=False)
                        else:
                            true_frame.loc[len(true_frame)] = [i + 1, stance_test[i], test_predict[i], tweets_test[i].text_raw]
                            true_frame.to_excel(True_file, encoding='utf-8', index=False)


                return favg,clf_model.best_score_,clf_model.best_estimator_,\
                       clf_model.cv_results_['std_test_score'][clf_model.best_index_], prec, recall, f, Conf,accuracy, diff_time,

            FAVG=[];CVscores=[];TUNED=[]

            from sklearn.model_selection import GridSearchCV


            if "rf" in clf_to_use or "all" in clf_to_use:
                #print("RandomForest")
                #estimator_file, test_score_file, cv_score_file=estimator_file_generate("rf",topn,skip_cbow,w2v_features,w2v_ctxt,w2v_min_wc,down_sampling,LDA_topics,LDA_passes)
                estimator_file, test_score_file, cv_score_file=estimator_file_generate("rf",skip_cbow,w2v_features,w2v_ctxt,w2v_min_wc,down_sampling)

                if os.path.isfile(test_score_file):
                    favg_rf = joblib.load(test_score_file)
                    if os.path.isfile(estimator_file):
                        tuned_rf = joblib.load(estimator_file)
                        if os.path.isfile(cv_score_file):
                            cv_rf = joblib.load(cv_score_file)
                            #print("favg_rf",favg_rf)
                            #print("cv_rf",cv_rf)
                            #print("tuned_rf",tuned_rf)
                else:


                    from sklearn.ensemble import RandomForestClassifier
                    if rf_weight=='balanced':
                        rf=RandomForestClassifier(random_state=R_Seed,n_jobs=-1,verbose=False,class_weight='balanced')
                    else:
                        rf=RandomForestClassifier(random_state=R_Seed,n_jobs=-1,verbose=False)


                    favg_rf,cv_rf,tuned_rf,std, prec, recall, f, Conf, accuracy, diff_time=clf_score(rf,param_grid_rf,stance_train,stance_test,None,'rf')

                    try:
                        pvalue_time = pvalue_time;
                    except:pvalue_time = 0
                    try:TOTAL_NUM_FEAT = TOTAL_NUM_FEAT;
                    except:TOTAL_NUM_FEAT = 0
                    try:NUM_FEAT = NUM_FEAT

                    except:
                        NUM_FEAT = len(feat)




                    MFF=tuned_rf.max_features
                    MDD=tuned_rf.max_depth
                    if MFF==None: MFF="None"
                    if MDD==None: MDD="None"



                    dframe_op_rf.loc[len(dframe_op_rf)] = [op, cv_rf, std, str(tuned_rf), tuned_rf.n_estimators, \
                                                           tuned_rf.min_samples_leaf, MFF, MDD, \
                                                           favg_rf, Conf, prec, recall, \
                                                           accuracy, f, diff_time,
                                                           NUM_FEAT]
                    dframe_op_rf.to_excel(xls_op_rf, encoding='utf-8', index=False)

                FAVG.append(favg_rf);CVscores.append(cv_rf);TUNED.append(tuned_rf);

            if "svm"  in clf_to_use or "all" in clf_to_use:
                #estimator_file, test_score_file, cv_score_file=estimator_file_generate("svm",topn,skip_cbow,w2v_features,w2v_ctxt,w2v_min_wc,down_sampling,LDA_topics,LDA_passes)
                estimator_file, test_score_file, cv_score_file=estimator_file_generate("svm",skip_cbow,w2v_features,w2v_ctxt,w2v_min_wc,down_sampling)

                if os.path.isfile(test_score_file):
                    #print ( "SVM")
                    favg_svm = joblib.load(test_score_file)
                    if os.path.isfile(estimator_file):
                        tuned_svm = joblib.load(estimator_file)
                        if os.path.isfile(cv_score_file):
                            cv_svm = joblib.load(cv_score_file)
                else:

                    from sklearn.svm import LinearSVC
                    #print("SVM")
                    svmclf=LinearSVC(random_state=R_Seed,max_iter=1000)


                    from time import time
                    start = time()
                    favg_svm, cv_svm, tuned_svm ,std, prec, recall, f, Conf, accuracy, diff_time = clf_score(svmclf, param_grid_svm,stance_train,stance_test,None,"svm")
                    #print("SVM  took %.2f seconds " % ((time() - start)))
                    try:
                        pvalue_time=pvalue_time;
                    except:pvalue_time = 0
                    try:TOTAL_NUM_FEAT=TOTAL_NUM_FEAT;
                    except:TOTAL_NUM_FEAT = 0
                    try:NUM_FEAT=NUM_FEAT
                    #except:NUM_FEAT = 0
                    except:
                        NUM_FEAT = len(feat)

                    #try:fs_time=fs_time
                    #except:fs_time=0


                    dframe_op_svm.loc[len(dframe_op_svm)] = [op, cv_svm, std, str(tuned_svm), tuned_svm.C, favg_svm,
                                                             Conf, prec, recall,
                                                             accuracy, f, diff_time, NUM_FEAT]

                    dframe_op_svm.to_excel(xls_op_svm, encoding='utf-8', index=False)

                FAVG.append(favg_svm);CVscores.append(cv_svm);TUNED.append(tuned_svm);

            if clf_to_use==['svm']:
                return favg_svm, cv_svm, tuned_svm

            elif clf_to_use==['rf']:
                return favg_rf, cv_rf, tuned_rf

            else:
                return None,None,None,None


        #def feature_selection_class(SPECIFIED_feat_cv_list,features_fine,LDA_topics,LDA_passes,w2v_features,w2v_min_wc,w2v_ctxt,down_sampling,skip_cbow,stance_train,stance_test,clf_to_use,N=None,method=None):
        def feature_selection_class(SPECIFIED_feat_cv_list, features_fine, w2v_features,
                                        w2v_min_wc, w2v_ctxt, down_sampling, skip_cbow, stance_train, stance_test,
                                        clf_to_use, N=None, method=None):

            try:
                if remove_w2v==True:

                    allfeatures.remove('word2vecmodel')
            except: pass


            if Google_w2v==True:
                w2v_features=None; w2v_min_wc=None; w2v_ctxt=None;down_sampling=None; skip_cbow=None;
            if twitter_w2v==True:
                w2v_features=None; w2v_min_wc=None; w2v_ctxt=None;down_sampling=None; skip_cbow=None;




            Results = computescores(stance_train, stance_test, allword2vec, w2v_features, w2v_min_wc, w2v_ctxt,
                                    down_sampling, skip_cbow, tweets_test, tweets_train, TARGET,
                                    allfeatures, favor_words, against_words, favor_tags,
                                    against_tags, clf_to_use, N, method)



            if clf_to_use==['svm']:
                favg_svm=Results[0]; cv_svm=Results[1]; tuned_svm=Results[2] ;
                try:
                    feat = Results[3]
                    return favg_svm, cv_svm, tuned_svm,feat
                except:
                    return favg_svm, cv_svm, tuned_svm

            elif clf_to_use==['rf']:
                favg_rf=Results[0]; cv_rf=Results[1]; tuned_rf=Results[2] ;
                try:
                    feat = Results[3];
                    return favg_rf, cv_rf, tuned_rf,feat
                except:
                    try:
                        return favg_rf, cv_rf, tuned_rf
                    except:
                        return None,None,None,None


            else:
                return None,None,None

        methods=["kbest_fclassif","kbest_chi","kbest_mutualinfo","featureset",None]


        remove_w2v=False


        allword2vec=False
        generate_allfeat=False


        if tune_clfs == True:


            if TARGET == "Climate Change is a Real Concern":
                param_grid_rf = {
                    'n_estimators': [5,10, 100, 200, 300, 400],
                    'max_features': [None, 0.9, 0.7, 0.5, 0.3],
                    'max_depth': [None, 10, 20],
                    "min_samples_leaf": [1, 2, 5, 10],}
            else:
                param_grid_rf = {
                    'n_estimators': [10,100,200,300,400],
                    'max_features': [None,0.9,0.7,0.5,0.3],
                    'max_depth': [None,10,20],
                    "min_samples_leaf": [1, 2,5,10],
                                 }

            param_grid_svm = {
                'C': [0.01, 0.1, 1, 10, 100],
                'dual': [False],
                'penalty': ['l2'],
                'fit_intercept': [True],
                'loss': ['hinge'],
                "multi_class": ["crammer_singer"]  #
            }



        else:

            param_grid_rf = {'criterion': ['gini'],
                             'n_estimators': [10],
                             'max_features': [None],
                             'max_depth': [None],
                             "min_samples_leaf": [1]
                             }

            param_grid_svm = {
                'C': [1],
                'dual': [False],
                'penalty': ['l2'],
                'fit_intercept': [True],
                'loss': ['hinge'], "multi_class": ["crammer_singer"]}



        RF_scores=[];SVM_scores=[];
        RF_cv=[];SVM_cv=[];
        RF_tuned=[];SVM_tuned=[];

        Best_FAVG_test=[]
        Best_CLF_test=[]

        #NUM_features =[100]
        #MIN_word_count=[2]
        #CONT_size=[2]
        #DOWN_sample=[1e-2]
        #SKIPcbow=[1]
        #K=[500]#

        if Google_w2v==True or twitter_w2v==True:
            remove_w2v = False

        FEAT=[]


        SPECIFIED_feat_cv_list = None;
        features_fine = None;
        N=kval;method=METHOD;

        if Google_w2v==True or twitter_w2v==True:
            w2v_features=None; w2v_min_wc=None; w2v_ctxt=None; down_sampling=None;skip_cbow=None;


        favg, cv, tuned = feature_selection_class(
            SPECIFIED_feat_cv_list, features_fine, w2v_features, w2v_min_wc, w2v_ctxt, down_sampling,
            skip_cbow, stance_train, stance_test, clf_to_use, N, method)


    if 'svm' in clf_to_use:
        print ("End of running SVM classifiers for the single models..... ")
        if TARGET == "Hillary Clinton":
            print("Results of the single models are saved as hc_svm_pval_0.05_cv5.xlsx in the folder containing single_models.py")
        elif TARGET == "Feminist Movement":
            print("Results of the single models are saved as fem_svm_pval_0.05_cv5.xlsx in the folder containing single_models.py")
        elif  TARGET == "Legalization of Abortion":
            print("Results of the single models are saved as ab_svm_pval_0.05_cv5.xlsx in the folder containing single_models.py")
        elif TARGET == "Atheism":
            print("Results of the single models are saved as ath_svm_pval_0.05_cv5.xlsx in the folder containing single_models.py")
        elif TARGET == "Climate Change is a Real Concern":
            print("Results of the single models are saved as clm_svm_pval_0.05_cv5.xlsx in the folder containing single_models.py")

    if 'rf' in clf_to_use:
        print ("End of running RF classifiers for the single models..... ")
        if TARGET == "Hillary Clinton":
            print("Results of the single models are saved as hc_rf_pval_0.05_cv5.xlsx in the folder containing single_models.py")
        elif TARGET == "Feminist Movement":
            print("Results of the single models are saved as fem_rf_pval_0.05_cv5.xlsx in the folder containing single_models.py")
        elif  TARGET == "Legalization of Abortion":
            print("Results of the single models are saved as ab_rf_pval_0.05_cv5.xlsx in the folder containing single_models.py")
        elif TARGET == "Atheism":
            print("Results of the single models are saved as ath_rf_pval_0.05_cv5.xlsx in the folder containing single_models.py")
        elif TARGET == "Climate Change is a Real Concern":
            print("Results of the single models are saved as clm_rf_pval_0.05_cv5.xlsx in the folder containing single_models.py")

