
import os.path
import numpy as np
import re
import nltk
import sklearn
from sklearn.externals import joblib
import pickle
from sklearn.metrics.classification import precision_recall_fscore_support, accuracy_score,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
import pandas as pd
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def compute_ensemble_test(TARGET):
    #TARGET ="Atheism"#"Feminist Movement"#"Hillary Clinton"#"Legalization of Abortion"##"Hillary Clinton"#"Feminist Movement"#"Climate Change is a Real Concern"#"Hillary Clinton"##"Legalization of Abortion"#"Feminist Movement"##"Atheism"#Feminist Movement"##"Atheism"#"#"Atheism"#
    #print("dataset ", TARGET)
    clfs=[['svm'],['rf']]#[['svm'],['rf'],['gnb']]
    cv=5
    pval=0.05
    write_separate_excels=False


    compute_error=False


    if TARGET=="Hillary Clinton":
        xl="hc_"; folder="hc/"
    elif TARGET=="Feminist Movement":
        xl="fem_";folder="feminist/"
    elif TARGET=="Climate Change is a Real Concern":
        xl="clm_";folder="climate/"
    elif TARGET=="Legalization of Abortion":
        xl="ab_";folder="abortion/"
    elif TARGET=="Atheism":
        xl="ath_";folder="atheism/"

    settings="_OnePhase.xlsx"
    nodicmodels = False
    if nodicmodels:
        MODELS = [4, 5, 6, 10, 11, 12]
    else:
        MODELS = range(1, 13)

    xlsfile_hard_rf=xl+'rf_hard_Test'+settings
    xlsfile_hard_svm=xl+'svm_hard_Test'+settings
    xlsfile_hard_svm_rf=xl+'svm_rf_hard_Test'+settings
    xlsfile_hard_total=xl+'Ensembles_Test.xlsx'


    TEST_vote=[]

    first_iter=True
    first_iter_rf=True
    first_iter_svm=True

    import pandas as pd

    columns = ["Test F-score", "Confusion", "Precision", "Recall", "Accuracy", "classes F-scores", "Time"]
    columns_total = ["Ensemble","Test F-score", "Confusion", "Precision", "Recall", "Accuracy", "classes F-scores", "Time"]

    try:
        dframe_hard_rf=pd.read_excel(xlsfile_hard_rf)
    except:
        dframe_hard_rf = pd.DataFrame(columns=columns)
    try:
        dframe_hard_svm=pd.read_excel(xlsfile_hard_svm)
    except:
        dframe_hard_svm = pd.DataFrame(columns=columns)

    try:
        dframe_hard_svm_rf=pd.read_excel(xlsfile_hard_svm_rf)
    except:
        dframe_hard_svm_rf = pd.DataFrame(columns=columns)

    try:
        dframe_hard_total=pd.read_excel(xlsfile_hard_total)
    except:
        dframe_hard_total = pd.DataFrame(columns=columns_total)



    PREC_hard=[];REC_hard=[];F_hard=[]
    ACC_hard=[];CONF_hard=[];FAVG_hard=[]



    title="Proposed_model"
    first_iter=True
    first_iter_rf = True
    first_iter_svm = True
    start1=time()
    for clf_to_use in clfs:
        if 'rf' in clf_to_use:
            print("  Running the tuned RF classifiers on the single models.... ")
        elif 'svm' in clf_to_use:
            print("  Running the tuned SVM classifiers on the single models.... ")

        start2=time()
        #print(clf_to_use)

        if clf_to_use==['rf']:
            first_iter_rf = True
        elif clf_to_use==['svm']:
            first_iter_svm = True

        for op in MODELS:
            print("     Model #",op)

            script_dir = os.path.dirname(__file__)




            if TARGET == "Hillary Clinton":
                rel_path = r"hc/" + "op" + str(op) + "/"
                stance_train=[2, 1, 2, 2, 0, 2, 0, 0, 2, 2, 2, 2, 1, 0, 2, 0, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 1, 2, 2, 0,
                 2, 2, 0, 2, 2, 2, 2, 0, 2, 1, 0, 1, 2, 1, 2, 0, 1, 0, 2, 1, 2, 0, 1, 0, 0, 2, 0, 2, 0, 2, 1, 1, 1, 0, 0, 1, 2,
                 0, 2, 2, 1, 2, 0, 1, 2, 1, 2, 1, 0, 1, 1, 1, 2, 0, 1, 1, 1, 2, 0, 2, 2, 1, 2, 2, 2, 0, 2, 2, 0, 1, 2, 0, 2, 0,
                 0, 2, 2, 0, 2, 2, 2, 0, 0, 2, 0, 0, 1, 0, 1, 2, 2, 0, 1, 1, 2, 2, 2, 1, 0, 0, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0,
                 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 2, 2, 2, 0, 1, 0, 1, 1, 2, 1, 1, 0, 1, 2, 2, 1, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 2,
                 2, 0, 2, 2, 1, 2, 0, 0, 1, 2, 2, 1, 0, 2, 1, 2, 0, 1, 2, 1, 1, 1, 1, 1, 1, 0, 2, 0, 2, 1, 1, 2, 0, 2, 1, 2, 0,
                 2, 2, 2, 1, 0, 1, 0, 2, 2, 0, 2, 2, 1, 0, 0, 2, 1, 2, 0, 2, 0, 0, 2, 0, 0, 2, 1, 2, 0, 2, 0, 0, 1, 0, 1, 2, 2,
                 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 2, 2, 1, 2, 2, 2, 0, 2, 2, 1, 0, 1, 2, 2, 1, 0, 2, 2, 2, 2, 1, 0, 2, 1, 0, 0,
                 0, 2, 1, 0, 0, 1, 1, 2, 2, 2, 2, 2, 1, 0, 2, 0, 2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 2, 2, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 1, 0, 2, 0, 1, 1, 2, 2, 2, 2, 2, 0, 0,
                 1, 1, 2, 2, 0, 2, 1, 0, 0, 0, 0, 2, 1, 2, 2, 2, 0, 1, 0, 2, 2, 0, 0, 2, 0, 2, 1, 0, 1, 2, 0, 2, 0, 2, 1, 1, 0,
                 2, 0, 2, 0, 0, 0, 0, 1, 1, 1, 0, 2, 1, 2, 0, 2, 2, 2, 0, 2, 0, 0, 0, 0, 2, 2, 1, 0, 0, 2, 0, 2, 0, 0, 2, 2, 2,
                 1, 1, 1, 2, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 0, 0,
                 1, 1, 2, 0, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 2, 2, 2, 1, 2, 0, 2, 0, 0, 0, 2, 0, 2, 2, 0, 0,
                 0, 0, 1, 2, 0, 1, 0, 2, 2, 0, 2, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                stance_test=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 1, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 0, 2, 0, 1, 1, 2, 1, 1, 2, 1, 1, 0, 0, 0, 2, 1, 0, 1, 2,
                 0, 0, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 0, 2, 1, 0, 1, 2, 1, 2, 2, 0, 1, 0, 0, 0, 2, 0, 2, 2, 2, 1, 2, 1, 2,
                 0, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 2, 0, 2, 1, 2, 1, 2, 0, 1, 1, 2, 2, 2, 0, 0, 2, 2, 2, 0, 2, 2, 2, 0, 0, 0, 2,
                 0, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 0, 1, 0, 2, 0, 2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 0, 1, 0, 2, 2, 0, 2, 2,
                 1, 2, 2, 1, 1, 2, 2, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,
                 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 0, 2, 0, 1, 2, 2, 0, 2, 0, 2, 2, 2, 0, 0, 2, 1, 0, 1, 2, 0]

            elif TARGET == "Atheism":
                rel_path = r"atheism/" + "op" + str(op) + "/"
                stance_train=[2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 1, 2, 2, 2, 2, 2, 0, 2, 1, 2, 2, 1, 0, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 0, 2,
                 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 2, 1, 1, 2, 1, 2, 2, 0, 1, 1, 0, 1, 2, 0, 1, 0, 2, 2, 2, 2, 2, 2, 1, 2, 2,
                 2, 1, 2, 0, 1, 1, 0, 2, 2, 1, 0, 2, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1,
                 2, 0, 0, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 0, 1, 2, 0, 2, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1, 2, 1, 1, 1, 2,
                 0, 2, 2, 0, 2, 0, 1, 2, 2, 0, 2, 1, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 1, 0, 0, 2, 0, 0, 2, 2,
                 0, 0, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 1, 2, 1, 0, 2, 0, 0, 0, 2, 0, 1, 2, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 2, 1, 1, 2, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 2, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 2,
                 2, 2, 1, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 2, 0, 1, 2, 2, 0, 0, 2, 2, 0, 1, 2, 0, 0, 0, 2, 0, 2, 2, 2, 2, 0, 1,
                 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 1, 1, 2, 1, 2, 1, 0, 1, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1, 1, 0, 1, 1, 2,
                 2, 0, 1, 0, 1, 2, 1, 2, 0, 1, 0, 2, 2, 2, 2, 0, 0, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2, 0, 0, 2, 1, 2, 1, 0, 1, 1, 2,
                 2, 1, 2, 1, 0, 2, 2, 1, 2, 0, 0, 2, 2, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2,
                 1, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 0, 0, 2, 2, 1, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                stance_test=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 2,
                 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 0, 2, 2, 2, 2, 0, 0, 2, 0, 0, 0, 0, 2, 2, 2, 1, 0, 0, 0, 2, 1, 2, 2, 2, 2,
                 2, 2, 0, 2, 2, 0, 2, 1, 2, 1, 1, 1, 2, 2, 1, 0, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1,
                 2, 1, 2, 2, 2, 2, 0, 2, 0, 1, 0, 2, 0, 2, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 1, 2, 1, 2, 1, 1, 0, 2, 2, 0, 2, 2, 2, 0, 2, 2, 1, 2, 1, 2]

            elif TARGET == "Feminist Movement":
                rel_path = r"feminist/" + "op" + str(op) + "/"
                stance_train=[1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
                 1, 0, 1, 2, 0, 1, 1, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0, 1, 0,
                 1, 2, 2, 2, 0, 1, 1, 1, 1, 0, 1, 0, 2, 2, 0, 2, 1, 2, 0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 0, 1, 0, 2, 1, 2, 2, 1, 0,
                 1, 2, 2, 2, 1, 0, 1, 1, 1, 2, 0, 1, 0, 1, 2, 2, 1, 2, 0, 2, 2, 1, 2, 2, 0, 1, 0, 2, 0, 0, 1, 2, 0, 2, 2, 2, 2,
                 2, 1, 2, 1, 1, 1, 0, 1, 0, 2, 2, 0, 0, 0, 1, 1, 1, 2, 0, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 0, 2, 1, 1, 1, 1, 1,
                 1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 2, 1, 0, 2, 0, 1, 0, 0, 1, 0, 1, 0, 0, 2, 1, 1, 1, 1, 1, 1, 0, 2, 1, 1, 1, 1, 1,
                 1, 1, 2, 0, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 2, 2, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, 0, 0, 2, 1, 2, 2, 0, 1,
                 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 0, 2, 2, 1, 0, 0, 0, 1, 0, 2, 2, 0, 1, 0, 2, 1,
                 2, 1, 1, 1, 1, 2, 0, 1, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 1, 2, 0, 2, 2, 0, 2, 0, 0, 1, 0, 1, 0, 0, 2, 0, 2, 1, 2,
                 0, 2, 2, 0, 1, 0, 1, 1, 0, 1, 2, 0, 1, 0, 2, 0, 2, 2, 2, 2, 1, 1, 0, 1, 0, 1, 0, 2, 0, 2, 1, 1, 1, 0, 1, 0, 2,
                 1, 0, 0, 0, 2, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 0, 1, 1, 0, 0,
                 1, 2, 0, 1, 1, 0, 2, 0, 0, 2, 1, 1, 1, 2, 0, 2, 0, 1, 2, 1, 1, 2, 2, 0, 2, 2, 0, 2, 0, 0, 2, 1, 1, 2, 2, 2, 0,
                 0, 2, 2, 1, 1, 1, 2, 0, 1, 0, 1, 0, 1, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 1, 0, 2, 2, 0, 2, 1, 0, 0,
                 2, 0, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                stance_test=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 0, 2, 2, 1, 1, 1, 2, 1, 0, 0, 2, 0, 1, 1, 0, 2, 0, 0, 2, 2, 1, 0, 2, 0, 2, 1, 2, 0, 0, 1, 0, 2, 2, 2,
                 2, 0, 1, 1, 0, 1, 1, 0, 2, 2, 0, 2, 1, 0, 2, 1, 2, 2, 2, 2, 0, 2, 1, 0, 2, 0, 2, 2, 2, 0, 1, 2, 1, 1, 2, 1, 2,
                 0, 2, 0, 2, 1, 0, 2, 0, 2, 2, 0, 0, 1, 0, 0, 0, 1, 1, 2, 1, 0, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1,
                 1, 2, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 0, 2, 2, 1, 2, 2, 2, 1, 1, 0, 1, 0, 2, 0, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1,
                 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2,
                 2, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 0, 2, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 2]

            elif TARGET == "Legalization of Abortion":
                rel_path = r'abortion/' + "op" + str(op) + "/"
                stance_train=[2, 1, 2, 2, 0, 2, 2, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 2, 2, 1, 1, 0, 0, 1, 1, 1, 2, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1,
                 0, 2, 2, 1, 1, 1, 2, 2, 0, 0, 2, 0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 0, 2, 0, 1, 2,
                 1, 2, 1, 1, 1, 1, 2, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 0, 2, 2, 0, 1, 0, 2, 2, 0, 2, 1, 1, 2, 2, 0, 2,
                 2, 2, 2, 0, 2, 0, 0, 2, 0, 2, 2, 2, 0, 0, 0, 0, 2, 2, 0, 1, 2, 2, 0, 0, 2, 2, 1, 0, 0, 2, 0, 1, 2, 0, 0, 0, 1,
                 0, 0, 2, 1, 2, 2, 2, 1, 1, 1, 2, 0, 2, 2, 0, 0, 2, 1, 0, 2, 1, 0, 1, 1, 0, 2, 0, 0, 2, 0, 1, 0, 2, 2, 0, 0, 0,
                 2, 2, 1, 1, 2, 0, 2, 0, 2, 2, 0, 0, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 0, 0, 2, 1, 2, 2, 2, 2, 2, 2,
                 2, 0, 2, 2, 0, 2, 0, 0, 2, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 2, 0, 0, 1, 0, 2, 1, 0, 2, 2, 1, 0, 2, 1, 2, 1, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 2, 0, 1, 2, 1, 2, 2,
                 2, 2, 2, 1, 0, 2, 2, 0, 0, 0, 2, 0, 2, 0, 0, 2, 2, 2, 0, 2, 0, 2, 0, 1, 2, 2, 2, 0, 0, 0, 0, 2, 1, 0, 2, 2, 0,
                 2, 0, 1, 2, 2, 2, 2, 2, 1, 0, 2, 1, 2, 2, 2, 2, 0, 2, 2, 1, 0, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 2,
                 0, 1, 0, 0, 2, 0, 2, 2, 1, 2, 2, 2, 0, 1, 0, 0, 2, 1, 2, 0, 0, 1, 2, 2, 2, 0, 2, 1, 1, 0, 1, 2, 2, 0, 2, 2, 0,
                 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 2, 2, 0, 0, 1, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 1, 0, 1, 2, 0, 0,
                 2, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2,
                 2, 0, 1, 2, 0, 2, 2, 2, 2, 0, 0, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 2, 2, 2, 1,
                 2, 1, 1, 2, 1, 2, 0, 2, 1, 0, 1, 1, 1, 0, 0, 2, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                stance_test=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 1, 1, 0, 2, 0, 0, 0, 0, 1, 2, 2, 0,
                 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 0, 1, 0, 0, 2, 1, 2, 2, 1, 1, 0, 2, 0, 2, 0, 1, 0, 1, 2, 0, 1, 0, 2, 1, 0, 1,
                 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 2, 2, 0, 2, 1, 2, 2, 0, 2, 2, 0, 2, 1, 2, 2, 1, 1, 0, 1,
                 1, 2, 0, 1, 2, 2, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2, 1, 2, 2, 1, 0, 2, 2, 0, 2, 2, 0, 0, 2, 0, 2, 2, 0, 1, 2, 2,
                 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1,
                 2, 1, 2, 0, 0, 1, 1, 2, 0, 1, 0, 2, 0, 0, 2, 2, 0, 2, 2, 2, 2]

            elif TARGET == "Climate Change is a Real Concern":
                rel_path = r'climate/' + "op" + str(op) + "/"
                stance_train=[1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 1, 0, 1, 1, 0, 1, 0, 0, 2, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                 2, 0, 1, 0, 1, 2, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0,
                 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1,
                 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1,
                 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0,
                 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 2, 0,
                 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 2, 2,
                 2, 2, 2, 1, 1, 0, 2, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
                stance_test=[0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
                 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 0,
                 0, 1, 0, 0, 1, 1, 0, 0, 1, 2, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                 1, 2, 1, 2, 1, 1, 1, 0, 1, 2, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]

            abs_file_path = os.path.join(script_dir, rel_path)
            pathfile = abs_file_path

            #trainfile=pathfile+"all_train_allfeatures_k_50_lda_None_20_None_w2v_None_None_None_None_None.pkl"
            #testfile=pathfile+"all_test_allfeatures_k_50_lda_None_20_None_w2v_None_None_None_None_None.pkl"
            #featfile=pathfile+"all_feat_allfeatures_k_50_lda_None_20_None_w2v_None_None_None_None_None.pkl"

            trainfile = pathfile + "extracted_features_train_matrix.pkl"
            testfile = pathfile + "extracted_features_test_matrix.pkl"
            featfile = pathfile + "extracted_features_names.pkl"

            #if os.path.isfile(trainfile):
            #        x_train = joblib.load(trainfile)
            #        x_test = joblib.load(testfile)
            #        feat = joblib.load(featfile)

            try:
                x_train = joblib.load(trainfile)
            except:
                print("extracted_features_train_matrix.pkl is not found in " + folder + "op" + str(op))
                print("You should run the single_models.py before evluating the ensemble models...")
                print("End of code")
                exit()
            try:
                x_test = joblib.load(testfile)
            except:
                print("extracted_features_test_matrix.pkl is not found in " + folder + "op" + str(op))
                print("You should run the single_models.py before evluating the ensemble models...")
                print("End of code")
                exit()

            try:
                feat = joblib.load(featfile)
            except:
                print("extracted_features_names.pkl is not found in " + folder + "op" + str(op))
                print("You should run the single_models.py before evluating the ensemble models...")
                print("End of code")
                exit()


            try:
                x_train=x_train.toarray()
                x_test=x_test.toarray()
            except:
                pass


            if TARGET== "Climate Change is a Real Concern":
                dframe_op_rf = pd.read_excel("clm_rf_pval_0.05_cv5.xlsx")
                dframe_op_svm = pd.read_excel("clm_svm_pval_0.05_cv5.xlsx")
            elif TARGET == "Feminist Movement":
                dframe_op_rf = pd.read_excel("fem_rf_pval_0.05_cv5.xlsx")
                dframe_op_svm = pd.read_excel("fem_svm_pval_0.05_cv5.xlsx")
            elif TARGET == "Atheism":
                dframe_op_rf = pd.read_excel("ath_rf_pval_0.05_cv5.xlsx")
                dframe_op_svm = pd.read_excel("ath_svm_pval_0.05_cv5.xlsx")
            elif TARGET == "Legalization of Abortion":
                dframe_op_rf = pd.read_excel("ab_rf_pval_0.05_cv5.xlsx")
                dframe_op_svm = pd.read_excel("ab_svm_pval_0.05_cv5.xlsx")
            elif TARGET == "Hillary Clinton":
                dframe_op_rf = pd.read_excel("hc_rf_pval_0.05_cv5.xlsx")
                dframe_op_svm = pd.read_excel("hc_svm_pval_0.05_cv5.xlsx")

            c = dframe_op_svm[(dframe_op_svm['Op model'] == op)]['C'].values[0]
            n_estimators = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Estimators'].values[0]
            min_samples_leaf = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Min Samples Leaf'].values[0]
            max_features = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Max Features'].values[0]
            max_depth = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Max Depth'].values[0]


            if clf_to_use==['rf']:

                if max_depth == 'None': max_depth = None
                if max_features == 'None': max_features = None

                clf_model = RandomForestClassifier(random_state=99, n_jobs=-1, verbose=False, max_depth=max_depth,
                                                   max_features=max_features, \
                                                   min_samples_leaf=min_samples_leaf, n_estimators=n_estimators)
            elif clf_to_use==['svm']:
                from sklearn.svm import LinearSVC
                clf_model = LinearSVC(C=c,random_state=99, dual=False, penalty='l2',fit_intercept=True,loss='hinge',multi_class='crammer_singer', max_iter=1000)

            def avg_score(xx, stance_train):
                prec, recall, f, support = precision_recall_fscore_support(xx, stance_train, labels=[2, 1, 0],
                                                                           beta=1)
                faverage = (f[0] + f[1]) * 0.5
                return faverage


            f1_scorer = make_scorer(avg_score, greater_is_better=True)


            clf_model.fit(x_train, stance_train)
            try:
                test_predict = clf_model.predict(x_test)
            except:
                test_predict = clf_model.predict(x_test.toarray())

            if 'svm' not in clf_to_use:
                test_predict1=clf_model.predict_proba(x_test)
            else:
                test_predict1=0


            if first_iter==True:
                TEST_prob=test_predict1
                TEST_vote=test_predict
                first_iter=False
            else:
                TEST_prob=TEST_prob+test_predict1
                TEST_vote=np.vstack((TEST_vote,test_predict))
            if clf_to_use == ['rf'] and first_iter_rf == True:
                TEST_prob_rf = test_predict1
                TEST_vote_rf = test_predict
                first_iter_rf = False
            elif clf_to_use == ['rf'] and first_iter_rf == False:
                TEST_prob_rf=TEST_prob_rf+test_predict1
                TEST_vote_rf=np.vstack((TEST_vote_rf,test_predict))
            if clf_to_use == ['svm'] and first_iter_svm == True:
                TEST_prob_svm = test_predict1
                TEST_vote_svm = test_predict
                first_iter_svm = False
            elif clf_to_use == ['svm'] and first_iter_svm == False:
                TEST_prob_svm=TEST_prob_svm+test_predict1
                TEST_vote_svm=np.vstack((TEST_vote_svm,test_predict))
        diff_time=time()-start2
        if clf_to_use == ['svm']:
            time_svm=diff_time
        if clf_to_use == ['rf']:
            time_rf=diff_time

    def compute_score(TEST_vote,dframe_hard,xlsfile_hard,diff_time,classifiers):
        LABELS=[]

        fav_ag_no = 0;
        fav_ag = 0;
        fav_no = 0;
        ag_no = 0
        for i in range(TEST_vote.shape[1]):
            label_0 = list(TEST_vote[:, i]).count(0)
            label_1 = list(TEST_vote[:, i]).count(1)
            label_2 = list(TEST_vote[:, i]).count(2)

            if TARGET == "Atheism":
                if label_2 == max(label_0, label_1, label_2):
                    label = 2
                elif label_0 == max(label_0, label_1, label_2):
                    label = 0
                elif label_1 == max(label_0, label_1, label_2):
                    label = 1
            elif TARGET == "Legalization of Abortion":
                if label_2 == max(label_0, label_1, label_2):
                    label = 2
                elif label_0 == max(label_0, label_1, label_2):
                    label = 0
                elif label_1 == max(label_0, label_1, label_2):
                    label = 1
            elif TARGET == "Hillary Clinton":
                if label_2 == max(label_0, label_1, label_2):
                    label = 2
                elif label_0 == max(label_0, label_1, label_2):
                    label = 0
                elif label_1 == max(label_0, label_1, label_2):
                    label = 1
            elif TARGET == "Climate Change is a Real Concern":
                if label_1 == max(label_0, label_1, label_2):
                    label = 1
                elif label_0 == max(label_0, label_1, label_2):
                    label = 0
                elif label_2 == max(label_0, label_1, label_2):
                    label = 2
            elif TARGET == "Feminist Movement":
                if label_2 == max(label_0, label_1, label_2):
                    label = 2
                elif label_1 == max(label_0, label_1, label_2):
                    label = 1
                elif label_0 == max(label_0, label_1, label_2):
                    label = 0

            LABELS.append(label)

        diff_time = time() - start1
        #print("Fitting model took %.2f seconds " % (diff_time))

        if compute_error==True and classifiers=="rf_svm":
            Error_file = xl+"_rfsvm_error.xlsx"
            True_file = xl+"_rfsvm_True.xlsx"
            columns = ["Tweet_ID", "TrueLabel", "Predicted","Tweet"]
            try:
                error_frame = pd.read_excel(Error_file)
            except:
                error_frame = pd.DataFrame(columns=columns)
            try:
                true_frame = pd.read_excel(True_file)
            except:
                true_frame = pd.DataFrame(columns=columns)
            wrong = []
            for i in range(len(LABELS)):
                if LABELS[i] != stance_test[i]:
                    wrong.append(i + 1)
                    error_frame.loc[len(error_frame)] = [i + 1, stance_test[i],LABELS[i],tweets_test[i].text_raw]
                    error_frame.to_excel(Error_file, encoding='utf-8', index=False)
                else:
                    true_frame.loc[len(true_frame)] = [i + 1, stance_test[i], LABELS[i], tweets_test[i].text_raw]
                    true_frame.to_excel(True_file, encoding='utf-8', index=False)
            #print("wrong_ids ", wrong)
            #print("Number of tweets classified wrong ", len(wrong))

        prec, recall, f, support = precision_recall_fscore_support(stance_test, LABELS, labels=[2, 1, 0],
                                                                   beta=1)
        Conf = confusion_matrix(stance_test, LABELS, labels=[2, 1, 0])
        #print("Confusion matrix ", Conf)
        #print("precision ", prec)
        #print("recall ", recall)
        #print("support ", support)
        accuracy = accuracy_score(stance_test, LABELS)
        #print ("accuracy ", accuracy)
        #print ("f ", f)
        #print ("favg test set : ", (f[0] + f[1]) * 0.5)
        favg = (f[0] + f[1]) * 0.5

        dframe_hard.loc[len(dframe_hard)] = [ favg, Conf, prec, recall, accuracy, f, diff_time]
        if write_separate_excels:
            dframe_hard.to_excel(xlsfile_hard, encoding='utf-8', index=False)

        if classifiers=='rf':
            dframe_hard_total.loc[len(dframe_hard_total)] = ['RF', favg, Conf, prec, recall, accuracy, f, diff_time]
            dframe_hard_total.to_excel(xlsfile_hard_total, encoding='utf-8', index=False)
        elif classifiers=='svm':
            dframe_hard_total.loc[len(dframe_hard_total)] = ['SVM', favg, Conf, prec, recall, accuracy, f, diff_time]
            dframe_hard_total.to_excel(xlsfile_hard_total, encoding='utf-8', index=False)
        elif classifiers=="rf_svm":
            dframe_hard_total.loc[len(dframe_hard_total)] = ['RF SVM', favg, Conf, prec, recall, accuracy, f, diff_time]
            dframe_hard_total.to_excel(xlsfile_hard_total, encoding='utf-8', index=False)

        PREC_hard.append(prec)
        REC_hard.append(recall)
        F_hard.append(f)
        ACC_hard.append(accuracy)
        CONF_hard.append(Conf)
        FAVG_hard.append(favg)
        return favg, Conf



    LABELS=[]; LABELS_rf=[]; LABELS_svm=[];
    LABELS_svm_rf=[];


    if ['rf'] in clfs:
        #print("RF ONLY.................")
        favg_rf, conf_rf=compute_score(TEST_vote_rf, dframe_hard_rf,xlsfile_hard_rf,time_rf,"rf")

    if ['svm'] in clfs:
        #print("SVM ONLY.............")
        favg_svm, conf_svm=compute_score(TEST_vote_svm, dframe_hard_svm,xlsfile_hard_svm,time_svm,"svm")
    if ['rf'] in clfs and ['svm'] in clfs:
        #print("RF and SVM BOTH.............")
        time_rf_svm=time_rf+time_svm
        TEST_vote_svm_rf = np.vstack((TEST_vote_svm, TEST_vote_rf))
        favg_svm_rf, conf_svm_rf=compute_score(TEST_vote_svm_rf, dframe_hard_svm_rf,xlsfile_hard_svm_rf, time_rf_svm,"rf_svm")
    return favg_svm, conf_svm, favg_rf, conf_rf, favg_svm_rf, conf_svm_rf

    #p=p+1
