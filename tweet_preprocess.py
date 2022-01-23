import sqlite3 as lite
import os.path
import numpy as np
import re
import nltk
#import sklearn
#import Features_manager_modified
from sklearn.externals import joblib
import pickle

#import os.open
from Tweet import make_tweet
#import gensim
#from sklearn.ensemble import VotingClassifier

from time import time

print("Running pre-processing for a dataset of your selection...")
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


train_or_test=["tweets_train","tweets_test"]
dictionary=[True, False]

TARGET_TEST=TARGET



for data in train_or_test:
    for include_dictionary in dictionary:
        if data=="tweets_test":
            print("Reading Test data from database ..... ")
        elif data=="tweets_train":
            print("Reading Training data from database......... ")
        if include_dictionary:
            print("Pre-processing while Including the manual dictionary.....")
        else:
            print("Pre-processing while Excluding the manual dictionary......")
        if include_dictionary:
            if TARGET =="Atheism":
                rel_path=r"atheism/newdic/"
            elif TARGET=="Hillary Clinton":
                rel_path=r"hc/newdic/"
            elif TARGET=="Climate Change is a Real Concern":
                rel_path = r"climate/newdic/"
            elif TARGET=="Feminist Movement":
                rel_path = r"feminist/newdic/"
            elif TARGET=="Legalization of Abortion":
                rel_path = r"abortion/newdic/"
        elif include_dictionary==False:
            if TARGET == "Atheism":
                rel_path = r"atheism/nodic/"
            elif TARGET == "Hillary Clinton":
                rel_path = r"hc/nodic/"
            elif TARGET == "Climate Change is a Real Concern":
                rel_path = r"climate/nodic/"
            elif TARGET == "Feminist Movement":
                rel_path = r"feminist/nodic/"
            elif TARGET == "Legalization of Abortion":
                rel_path = r"abortion/nodic/"
        if TARGET == "Atheism":
            folder = "atheism"
        elif TARGET == "Hillary Clinton":
            folder = "hc"
        elif TARGET == "Climate Change is a Real Concern":
            folder = "climate"
        elif TARGET == "Feminist Movement":
            folder = "feminist"
        elif TARGET == "Legalization of Abortion":
            folder = "abortion"

        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        abs_file_path = os.path.join(script_dir, rel_path)
        pathfile = abs_file_path



        def return_sentiment(sent):
            label_sent=0
            if sent== "neg":
                label_sent=-1
            elif sent== "pos" :
                label_sent=1
            elif sent== "other":
                label_sent=0
            else :label_sent=2
            return label_sent


        def return_opinion(opin):
            label_opin=0
            if opin== "1.  The tweet explicitly expresses opinion about the target, a part of the target, or an aspect of the target.":
                label_opin=1
            elif opin== "2. The tweet does NOT expresses opinion about the target but it HAS opinion about something or someone other than the target.":
                label_opin=2
            elif opin== "3.  The tweet is not explicitly expressing opinion. (For example, the tweet is simply giving information.)":
                label_opin=3

            return label_opin

        def return_stance(stance):
            label_stance=0
            if stance== "AGAINST":
                label_stance=2
            elif stance=="FAVOR":
                label_stance=1
            elif stance=="NONE":
                label_stance=0

            return label_stance



        db = lite.connect("Stance.db")
        cur = db.cursor()
        db.commit()
        tweets = []
        against = []
        none = []
        favor = []



        filter=TARGET

        if filter != "none":
            if data == "tweets_train":
                cur.execute("SELECT ID, Target, Tweet, Stance,Opinion,Sentiment FROM tweets_train where Target=? order by ID",
                            [filter])
            else:
                cur.execute("SELECT ID,Target, Tweet, Stance,Opinion,Sentiment FROM tweets_test where Target=? order by ID",

                            [filter])

        else:
            if data == "tweets_train":
                cur.execute("SELECT ID, Target,Tweet,Stance,Opinion,Sentiment FROM tweets_train order by ID")
            else:
                cur.execute("SELECT ID,Target,Tweet,Stance,Opinion,Sentiment FROM tweets_test order by ID")


        alltweetsfile=pathfile+data+".npy"
        alltweets=cur.fetchall()

        #print(len(alltweets))



        i=0
        a=0
        b=10000
        for tweet in alltweets:

            if i >=a and i<b:

                TWEET=[]
                sentiment = return_sentiment(tweet[5])
                opiniontowards = return_opinion(tweet[4])
                stance = return_stance(tweet[3])
                this_tweet = make_tweet(include_dictionary, tweet[0], tweet[2], tweet[1], opiniontowards, sentiment, stance)
                tweetfile = pathfile + data+"_"+str(i)+".npy"
                TWEET.append(this_tweet)
                TWEET=np.array(TWEET)
                try:
                    np.save(tweetfile,TWEET)
                except:
                    if include_dictionary:
                        print("To save pre-processed tweets, create a folder called newdic in the folder "+folder+", then run again the code")
                    else:
                        print("To save pre-processed tweets, create a folder called nodic in the folder "+folder+", then run again the code")
                    exit()
                tweets.append(this_tweet)
                if stance == 2:
                    against.append(this_tweet)
                elif stance == 1:
                    favor.append(this_tweet)
                elif stance == 0:
                    none.append(this_tweet)
                else:
                    print("STANCE NOT KNOWN")
            if i>b:
                break
            i=i+1


        tweets=[]
        favor=[];against=[];none=[]
        for i in range(0, 10000):
            tweetfile = pathfile + data + "_" + str(i) + ".npy"
            try:
                this_tweet=np.load(tweetfile)
            except:
                break
            this_tweet=this_tweet[0]
            tweets.append(this_tweet)
            if this_tweet.stance == 2:
                against.append(this_tweet)
            elif this_tweet.stance == 1:
                favor.append(this_tweet)
            elif this_tweet.stance == 0:
                none.append(this_tweet)
            else:
                print("STANCE NOT KNOWN")

        if data=="tweets_train":
            np.save(pathfile + 'tweets_train' + filter + '.npy', tweets)
            np.save(pathfile + 'tweets_train_favor' + filter + '.npy', favor)
            np.save(pathfile + 'tweets_train_against' + filter + '.npy', against)
            np.save(pathfile + 'tweets_train_none' + filter + '.npy', none)
        else:
            np.save(pathfile + 'tweets_test' + filter + '.npy', tweets)
            np.save(pathfile + 'tweets_test_favor' + filter + '.npy', favor)
            np.save(pathfile + 'tweets_test_against' + filter + '.npy', against)
            np.save(pathfile + 'tweets_test_none' + filter + '.npy', none)

        if data=="tweets_train":
            print("  Pre-processed training tweets are saved into " + pathfile)
        elif data=="tweets_test":
            print("  Pre-processed test tweets are saved into "+pathfile)
