
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
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
import pandas as pd

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def compute_ensemble_cv(TARGET):
    #TARGET =d #"Atheism"#"Climate Change is a Real Concern"###"#####"
    clfs=[['rf'],['svm']]
    write_separate_excels=False
    cv=5
    pval=0.05


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
    nodicmodels=False
    if nodicmodels:
        MODELS=[4,5,6,10,11,12]
    else:
        MODELS=range(1,13)


    xlsfile_hard_rf=xl+'rf_hard_CV'+settings
    xlsfile_hard_svm=xl+'svm_hard_CV'+settings
    xlsfile_hard_svm_rf=xl+'svm_rf_hard_CV'+settings
    xlsfile_hard_total=xl+'Ensembles_CV.xlsx'


    #columns = ["Models","CV F-score", "std", "Precision folds", "Recall folds", "F-scores folds", "Accuracy folds","Confusion folds", "classes F-scores folds" , "Time"]
    columns = ["CV F-score", "std", "Precision folds", "Recall folds", "F-scores folds", "Accuracy folds", \
               "Confusion folds", "classes F-scores folds", "Time"]
    columns_total = ["Ensemble","CV F-score", "std", "Precision folds", "Recall folds", "F-scores folds", "Accuracy folds", \
               "Confusion folds", "classes F-scores folds", "Time"]
    try:
        dframe_hard_rf = pd.read_excel(xlsfile_hard_rf)
    except:
        dframe_hard_rf = pd.DataFrame(columns=columns)
    try:
        dframe_hard_svm = pd.read_excel(xlsfile_hard_svm)
    except:
        dframe_hard_svm = pd.DataFrame(columns=columns)

    try:
        dframe_hard_svm_rf = pd.read_excel(xlsfile_hard_svm_rf)
    except:
        dframe_hard_svm_rf = pd.DataFrame(columns=columns)

    try:
        dframe_hard_total = pd.read_excel(xlsfile_hard_total)
    except:
        dframe_hard_total = pd.DataFrame(columns=columns_total)


    TEST_vote=[]
    first_iter_fold1 =True
    first_iter_fold2 =True
    first_iter_fold3 =True
    first_iter_fold4 =True
    first_iter_fold5 =True
    first_iter_fold1_rf =True
    first_iter_fold2_rf =True
    first_iter_fold3_rf =True
    first_iter_fold4_rf =True
    first_iter_fold5_rf =True
    first_iter_fold1_svm =True
    first_iter_fold2_svm =True
    first_iter_fold3_svm =True
    first_iter_fold4_svm =True
    first_iter_fold5_svm =True



    PREC_hard=[];REC_hard=[];F_hard=[]
    ACC_hard=[];CONF_hard=[];FAVG_hard=[]

    CVSCORE_hard=[];
    STD_hard=[];



    first_iter_fold1 = True;first_iter_fold2 = True;first_iter_fold3 = True;first_iter_fold4 = True;first_iter_fold5 = True
    first_iter_fold1_rf = True;first_iter_fold2_rf = True;first_iter_fold3_rf = True;first_iter_fold4_rf = True;first_iter_fold5_rf = True
    first_iter_fold1_svm = True;first_iter_fold2_svm = True;first_iter_fold3_svm = True;first_iter_fold4_svm = True;first_iter_fold5_svm = True

    for clf_to_use in clfs:
        if 'rf' in clf_to_use:
            print("  Running the tuned RF classifiers on the single models.... ")
        elif 'svm' in clf_to_use:
            print("  Running the tuned SVM classifiers on the single models.... ")

        start1 = time()
        #print(clf_to_use)

        for op in MODELS:
            print("     Model #",op)
            #print("OP model ", op)
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

            trainfile = pathfile + "extracted_features_train_matrix.pkl"
            testfile = pathfile +  "extracted_features_test_matrix.pkl"
            featfile = pathfile +  "extracted_features_names.pkl"


            try:
                    x_train = joblib.load(trainfile)
            except:
                print("extracted_features_train_matrix.pkl is not found in "+ folder +"op"+str(op))
                print("You should run the single_models.py before evluating the ensemble models...")
                print("End of code")
                exit()
            try:
                    x_test = joblib.load(testfile)
            except:
                print("extracted_features_test_matrix.pkl is not found in "+folder+"op"+str(op))
                print("You should run the single_models.py before evluating the ensemble models...")
                print("End of code")
                exit()

            try:
                    feat = joblib.load(featfile)
            except:
                print("extracted_features_names.pkl is not found in "+folder+"op"+str(op))
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

            # if TARGET== "Climate Change is a Real Concern":
            #
            #
            #     if samefeatures_thesis==False:
            #         #print("samefeatures_thesis is False..............")
            #         c = 0.01
            #
            #         dframe_op_rf = pd.read_excel("clm_rf_pval_0.05_cv5.xlsx")
            #         n_estimators = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Estimators'].values[0]
            #         min_samples_leaf = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Min Samples Leaf'].values[0]
            #         max_features = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Max Features'].values[0]
            #         max_depth = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Max Depth'].values[0]
            #     else:
            #         c = 0.01
            #         if op == 1:
            #             max_depth = 10;max_features = 0.5;min_samples_leaf = 2;n_estimators = 100
            #         elif op == 2:
            #             max_depth = None;max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op == 3:
            #             max_depth = 20;max_features = 0.5;min_samples_leaf = 1; n_estimators = 200
            #         elif op == 4:
            #             max_depth = 10; max_features = 0.5;min_samples_leaf = 2;n_estimators = 100
            #         elif op == 5:
            #             max_depth = None;max_features = None;min_samples_leaf = 2;n_estimators = 10
            #         elif op == 6:
            #             max_depth = None;max_features = None;min_samples_leaf = 2; n_estimators = 10
            #         elif op == 7:
            #             max_depth = 20;max_features = 0.7;min_samples_leaf = 2;n_estimators = 5
            #         elif op == 8:
            #             max_depth = None;max_features = None;min_samples_leaf = 1;n_estimators = 200
            #         elif op == 9:
            #             max_depth = None;max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op == 10:
            #             max_depth = None;max_features = None;min_samples_leaf = 2;n_estimators = 10
            #         elif op == 11:
            #             max_depth = None;max_features = None;min_samples_leaf = 1;n_estimators = 200
            #         elif op == 12:
            #             max_depth = None;max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op == 13:
            #             max_depth = 20;max_features = 0.7;min_samples_leaf = 2;n_estimators = 5
            #         elif op == 14:
            #             max_depth = None;max_features = None;min_samples_leaf = 2;n_estimators = 10
            #         elif op == 15:
            #             max_depth = None;max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op == 16:
            #             max_depth = 20;max_features = 0.7;min_samples_leaf = 2;n_estimators = 5
            #         elif op == 17:
            #             max_depth = None;max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op == 18:
            #             max_depth = None;max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op == 19:
            #             max_depth = 20;max_features = 0.7;min_samples_leaf = 2;n_estimators = 5
            #         elif op == 20:
            #             max_depth = 20;max_features = 0.5;min_samples_leaf = 1;n_estimators = 200
            #         elif op == 21:
            #             max_depth = 20;max_features = 0.7;min_samples_leaf = 2;n_estimators = 5
            #         elif op == 22:
            #             max_depth = 20;max_features = 0.7;min_samples_leaf = 2;n_estimators = 5
            #         elif op == 23:
            #             max_depth = 20;max_features = 0.7;min_samples_leaf = 2;n_estimators = 5
            #         elif op == 24:
            #             max_depth = None;max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #
            #
            # elif TARGET=="Feminist Movement":
            #
            #     if samefeatures_thesis==False:
            #         #print("samefeatures_thesis is False..............")
            #         c=0.01
            #         dframe_op_rf = pd.read_excel("fem_rf_pval_0.05_cv5.xlsx")
            #         n_estimators = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Estimators'].values[0]
            #         min_samples_leaf = \
            #         dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Min Samples Leaf'].values[0]
            #         max_features = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Max Features'].values[0]
            #         max_depth = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Max Depth'].values[0]
            #     else:
            #         c = 0.01
            #         if op == 1:
            #             max_depth = 20;
            #             max_features = 0.3;
            #             min_samples_leaf = 1;
            #             n_estimators = 400;
            #         elif op == 2:
            #             max_depth = 20;
            #             max_features = 0.3;
            #             min_samples_leaf = 1;
            #             n_estimators = 400
            #         elif op == 3:
            #             max_depth = None;
            #             max_features = 0.7;
            #             min_samples_leaf = 1;
            #             n_estimators = 300;
            #         elif op == 4:
            #             max_depth = None;
            #             max_features = 0.3;
            #             min_samples_leaf = 5;
            #             n_estimators = 10
            #         elif op == 5:
            #             max_depth = 20;
            #             max_features = 0.3;
            #             min_samples_leaf = 1;
            #             n_estimators = 400
            #         elif op == 6:
            #             max_depth = 10;
            #             max_features = 0.7;
            #             min_samples_leaf = 2;
            #             n_estimators = 300;
            #         elif op == 7:
            #             max_depth = 10;
            #             max_features = 0.7;
            #             min_samples_leaf = 2;
            #             n_estimators = 300
            #         elif op == 8:
            #             max_depth = 20;
            #             max_features = 0.3;
            #             min_samples_leaf = 1;
            #             n_estimators = 400
            #         elif op == 9:
            #             max_depth = 10;
            #             max_features = 0.7;
            #             min_samples_leaf = 2;
            #             n_estimators = 300;
            #         elif op == 10:
            #             max_depth = 10;
            #             max_features = 0.7;
            #             min_samples_leaf = 2;
            #             n_estimators = 300
            #         elif op == 11:
            #             max_depth = 20;
            #             max_features = 0.5;
            #             min_samples_leaf = 5;
            #             n_estimators = 100;
            #         elif op == 12:
            #             max_depth = None;
            #             max_features = None;
            #             min_samples_leaf = 2;
            #             n_estimators = 200;
            #         elif op == 13:
            #             max_depth = None;
            #             max_features = 0.3;
            #             min_samples_leaf = 5;
            #             n_estimators = 10;
            #         elif op == 14:
            #             max_depth = 20;
            #             max_features = 0.3;
            #             min_samples_leaf = 1;
            #             n_estimators = 400;
            #         elif op == 15:
            #             max_depth = None;
            #             max_features = None;
            #             min_samples_leaf = 1;
            #             n_estimators = 200
            #         elif op == 16:
            #             max_depth = 20;
            #             max_features = 0.3;
            #             min_samples_leaf = 1;
            #             n_estimators = 400
            #         elif op == 17:
            #             max_depth = 20;
            #             max_features = 0.3;
            #             min_samples_leaf = 1;
            #             n_estimators = 400
            #         elif op == 18:
            #             max_depth = None;
            #             max_features = 0.7;
            #             min_samples_leaf = 2;
            #             n_estimators = 400
            #         elif op == 19:
            #             max_depth = None;
            #             max_features = 0.7;
            #             min_samples_leaf = 2;
            #             n_estimators = 400;
            #         elif op == 20:
            #             max_depth = 20;
            #             max_features = 0.3;
            #             min_samples_leaf = 1;
            #             n_estimators = 400
            #         elif op == 21:
            #             max_depth = None;
            #             max_features = None;
            #             min_samples_leaf = 1;
            #             n_estimators = 200;
            #         elif op == 22:
            #             max_depth = None;
            #             max_features = 0.7;
            #             min_samples_leaf = 1;
            #             n_estimators = 300
            #         elif op == 23:
            #             max_depth = 20;
            #             max_features = 0.3;
            #             min_samples_leaf = 1;
            #             n_estimators = 400;
            #         elif op == 24:
            #             max_depth = 20;
            #             max_features = 0.3;
            #             min_samples_leaf = 1;
            #             n_estimators = 400;
            #
            # elif TARGET=="Atheism":
            #     if samefeatures_thesis==False:
            #         #print("samefeatures_thesis is False..............")
            #         c = 0.01
            #
            #         dframe_op_rf = pd.read_excel("ath_rf_pval_0.05_cv5.xlsx")
            #         n_estimators = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Estimators'].values[0]
            #         min_samples_leaf = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Min Samples Leaf'].values[0]
            #         max_features = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Max Features'].values[0]
            #         max_depth = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Max Depth'].values[0]
            #     else:
            #         c=0.01
            #         if op==1:
            #             max_depth = 20; max_features = None;min_samples_leaf = 5;n_estimators = 400
            #         elif op==2:
            #             max_depth = 20; max_features = 0.5; min_samples_leaf = 5;n_estimators = 100;
            #         elif op==3:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 1;n_estimators = 300
            #         elif op==4:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op==5:
            #             max_depth = 20; max_features = 0.5;min_samples_leaf = 5;n_estimators = 100;
            #         elif op==6:
            #             max_depth = None; max_features = None;min_samples_leaf = 1;n_estimators = 200
            #         elif op==7:
            #             max_depth = None; max_features = None;min_samples_leaf = 2;n_estimators = 200
            #         elif op==8:
            #             max_depth = 20; max_features = 0.5;min_samples_leaf = 5;n_estimators = 100
            #         elif op==9:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 1;n_estimators = 300
            #         elif op==10:
            #             max_depth = 20; max_features = None;min_samples_leaf = 5;n_estimators = 100
            #         elif op==11:
            #             max_depth = None; max_features = None;min_samples_leaf = 5;n_estimators = 10
            #         elif op==12:
            #             max_depth = 10; max_features = 0.7;min_samples_leaf = 2;n_estimators = 300;
            #         elif op==13:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400;
            #         elif op==14:
            #             max_depth = 20; max_features = 0.5;min_samples_leaf = 5;n_estimators = 100;
            #         elif op==15:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 1;n_estimators = 300
            #         elif op==16:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op==17:
            #             max_depth = 10; max_features = 0.7;min_samples_leaf = 2;n_estimators = 300
            #         elif op==18:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 1;n_estimators = 300
            #         elif op==19:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op==20:
            #             max_depth = 10; max_features = 0.7;min_samples_leaf = 2;n_estimators = 300
            #         elif op==21:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 2;n_estimators = 400;
            #         elif op==22:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op==23:
            #             max_depth = 20; max_features = 0.5;min_samples_leaf = 5;n_estimators = 100
            #         elif op==24:
            #             max_depth = 10; max_features = 0.7;min_samples_leaf = 2;n_estimators = 300;
            #
            #
            # elif TARGET=="Legalization of Abortion":
            #     if samefeatures_thesis==False:
            #         c=0.01
            #         if op==2: c=0.1
            #         dframe_op_rf = pd.read_excel("ab_rf_pval_0.05_cv5.xlsx")
            #         n_estimators = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Estimators'].values[0]
            #         min_samples_leaf = \
            #         dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Min Samples Leaf'].values[0]
            #         max_features = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Max Features'].values[0]
            #         max_depth = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Max Depth'].values[0]
            #     else:
            #         c=0.01
            #         if op==1:
            #             max_depth = 20; max_features = 0.3; min_samples_leaf = 1;n_estimators = 400;
            #         elif op==2:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op==3:
            #             c=0.1
            #             max_depth = None; max_features = None; min_samples_leaf = 2;n_estimators = 200
            #         elif op==4:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400;
            #         elif op==5:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400;
            #         elif op==6:
            #             c=0.1
            #             max_depth = None; max_features = None;min_samples_leaf = 1;n_estimators = 200
            #         elif op==7:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op==8:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op==9:
            #             max_depth = None;max_features = None;min_samples_leaf = 1;n_estimators = 200;
            #         elif op==10:
            #             max_depth = 20; max_features = 0.5;min_samples_leaf = 5;n_estimators = 100;
            #         elif op==11:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op==12:
            #             max_depth = None; max_features = None;min_samples_leaf = 1;n_estimators = 200
            #         elif op==13:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 2;n_estimators = 400
            #         elif op==14:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op==15:
            #             max_depth = None; max_features = None;min_samples_leaf = 1;n_estimators = 200
            #         elif op==16:
            #             max_depth = None; max_features = 0.3;min_samples_leaf = 5;n_estimators = 10
            #         elif op==17:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op==18:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400;
            #         elif op==19:
            #             max_depth = None; max_features = 0.3;min_samples_leaf = 5;n_estimators = 10
            #         elif op==20:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400;
            #         elif op==21:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 1;n_estimators = 300
            #         elif op==22:
            #             max_depth = None; max_features = 0.3;min_samples_leaf = 5;n_estimators = 10
            #         elif op==23:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op==24:
            #             max_depth = None; max_features = None;min_samples_leaf = 1;n_estimators=200
            #
            # elif TARGET=="Hillary Clinton":
            #     if samefeatures_thesis==False:
            #         c=0.01
            #         dframe_op_rf = pd.read_excel("hc_rf_pval_0.05_cv5.xlsx")
            #         n_estimators = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Estimators'].values[0]
            #         min_samples_leaf = \
            #         dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Min Samples Leaf'].values[0]
            #         max_features = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Max Features'].values[0]
            #         max_depth = dframe_op_rf[(dframe_op_rf['Op model'] == op)]['Max Depth'].values[0]
            #     else:
            #         c=0.01
            #         if op==1:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 2;n_estimators = 400;
            #         elif op==2:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 1;n_estimators = 300;
            #         elif op==3:
            #             max_depth = 20; max_features = None;min_samples_leaf = 5;n_estimators = 400
            #         elif op==4:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 1;n_estimators = 300
            #         elif op==5:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 1;n_estimators = 300
            #         elif op==6:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 1;n_estimators = 300;
            #         elif op==7:
            #             max_depth = None; max_features = None;min_samples_leaf = 1;n_estimators = 200;
            #         elif op==8:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400;
            #         elif op==9:
            #             max_depth = 10; max_features = 0.9;min_samples_leaf = 10;n_estimators = 200
            #         elif op==10:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 1;n_estimators = 300
            #         elif op==11:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op==12:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 1;n_estimators = 300;
            #         elif op==13:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 2;n_estimators = 400;
            #         elif op==14:
            #             max_depth = None; max_features = None;min_samples_leaf = 1;n_estimators = 200;
            #         elif op==15:
            #             max_depth = 10; max_features = 0.9;min_samples_leaf = 10;n_estimators = 200;
            #         elif op==16:
            #             max_depth = 10; max_features = 0.7;min_samples_leaf = 2;n_estimators = 300
            #         elif op==17:
            #             max_depth = 20; max_features = 0.3;min_samples_leaf = 1;n_estimators = 400
            #         elif op==18:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 2;n_estimators = 400
            #         elif op==19:
            #             max_depth = None; max_features = None;min_samples_leaf = 1;n_estimators = 200;
            #         elif op==20:
            #             max_depth = None; max_features = None;min_samples_leaf = 2;n_estimators = 200;
            #         elif op==21:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 2;n_estimators = 400
            #         elif op==22:
            #             max_depth = None; max_features = None;min_samples_leaf = 1;n_estimators = 200;
            #         elif op==23:
            #             max_depth = 10; max_features = 0.9;min_samples_leaf = 10;n_estimators = 200
            #         elif op==24:
            #             max_depth = None; max_features = 0.7;min_samples_leaf = 1;n_estimators = 300


            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=cv, random_state=99, shuffle=True)
            X = x_train
            y = np.array(stance_train)
            Fs = []
            #print("Splitting train items .... ")
            j=1
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = y[train_index], y[test_index]

                if clf_to_use==['rf']:
                    if max_depth=='None': max_depth=None
                    if max_features=='None': max_features=None

                    clf_model= RandomForestClassifier(random_state=99, n_jobs=-1, verbose=False, max_depth=max_depth,  max_features= max_features, \
                                        min_samples_leaf=min_samples_leaf, n_estimators=n_estimators)
                elif clf_to_use == ['svm']:
                    from sklearn.svm import LinearSVC
                    clf_model = LinearSVC(C=c, random_state=99, dual=False, penalty='l2', fit_intercept=True, loss='hinge',
                                          multi_class='crammer_singer', max_iter=1000)


                clf_model.fit(X_train, Y_train)

                try:
                    test_predict = clf_model.predict(X_test)
                except:
                    test_predict = clf_model.predict(X_test.toarray())

                if 'svm' not in clf_to_use:
                    test_predict1=clf_model.predict_proba(X_test)
                else:
                    test_predict1=0


                if j==1: #fold 1
                    if first_iter_fold1 == True: #fisrt model
                        TEST_A_prob = test_predict1
                        TEST_A_vote = test_predict
                        test_A=Y_test#SAVE true labels for this fold
                        first_iter_fold1  = False
                    else:
                        TEST_A_prob = TEST_A_prob + test_predict1
                        TEST_A_vote = np.vstack((TEST_A_vote, test_predict))
                    if clf_to_use == ['rf'] and first_iter_fold1_rf == True:
                        TEST_A_prob_rf = test_predict1
                        TEST_A_vote_rf = test_predict
                        first_iter_fold1_rf = False
                    elif clf_to_use == ['rf'] and first_iter_fold1_rf == False:
                        TEST_A_prob_rf = TEST_A_prob_rf + test_predict1
                        TEST_A_vote_rf = np.vstack((TEST_A_vote_rf, test_predict))

                    if clf_to_use == ['svm'] and first_iter_fold1_svm == True:
                        TEST_A_prob_svm = test_predict1
                        TEST_A_vote_svm = test_predict
                        first_iter_fold1_svm = False
                    elif clf_to_use == ['svm'] and first_iter_fold1_svm == False:
                        TEST_A_prob_svm = TEST_A_prob_svm + test_predict1
                        TEST_A_vote_svm = np.vstack((TEST_A_vote_svm, test_predict))
                elif j==2:#fold 2
                    if first_iter_fold2 == True:
                        TEST_B_prob = test_predict1
                        TEST_B_vote = test_predict
                        test_B=Y_test#SAVE true labels for this fold
                        first_iter_fold2 = False
                    else:
                        TEST_B_prob = TEST_B_prob + test_predict1
                        TEST_B_vote = np.vstack((TEST_B_vote, test_predict))
                    if clf_to_use == ['rf'] and first_iter_fold2_rf == True:
                        TEST_B_prob_rf = test_predict1
                        TEST_B_vote_rf = test_predict
                        first_iter_fold2_rf = False
                    elif clf_to_use == ['rf'] and first_iter_fold2_rf == False:
                        TEST_B_prob_rf = TEST_B_prob_rf + test_predict1
                        TEST_B_vote_rf = np.vstack((TEST_B_vote_rf, test_predict))

                    if clf_to_use == ['svm'] and first_iter_fold2_svm == True:
                        TEST_B_prob_svm = test_predict1
                        TEST_B_vote_svm = test_predict
                        first_iter_fold2_svm = False
                    elif clf_to_use == ['svm'] and first_iter_fold2_svm == False:
                        TEST_B_prob_svm = TEST_B_prob_svm + test_predict1
                        TEST_B_vote_svm = np.vstack((TEST_B_vote_svm, test_predict))
                elif j==3: #fold 3
                    if first_iter_fold3  == True:
                        TEST_C_prob = test_predict1
                        TEST_C_vote = test_predict
                        test_C=Y_test#SAVE true labels for this fold
                        first_iter_fold3  = False
                    else:
                        TEST_C_prob = TEST_C_prob + test_predict1
                        TEST_C_vote = np.vstack((TEST_C_vote, test_predict))
                    if clf_to_use == ['rf'] and first_iter_fold3_rf == True:
                        TEST_C_prob_rf = test_predict1
                        TEST_C_vote_rf = test_predict
                        first_iter_fold3_rf = False
                    elif clf_to_use == ['rf'] and first_iter_fold3_rf == False:
                        TEST_C_prob_rf = TEST_C_prob_rf + test_predict1
                        TEST_C_vote_rf = np.vstack((TEST_C_vote_rf, test_predict))

                    if clf_to_use == ['svm'] and first_iter_fold3_svm == True:
                        TEST_C_prob_svm = test_predict1
                        TEST_C_vote_svm = test_predict
                        first_iter_fold3_svm = False
                    elif clf_to_use == ['svm'] and first_iter_fold3_svm == False:
                        TEST_C_prob_svm = TEST_C_prob_svm + test_predict1
                        TEST_C_vote_svm = np.vstack((TEST_C_vote_svm, test_predict))
                elif j==4:#fold 4
                    if first_iter_fold4 == True:
                        TEST_D_prob = test_predict1
                        TEST_D_vote = test_predict
                        test_D=Y_test#SAVE true labels for this fold
                        first_iter_fold4 = False
                    else:
                        TEST_D_prob = TEST_D_prob + test_predict1
                        TEST_D_vote = np.vstack((TEST_D_vote, test_predict))
                    if clf_to_use == ['rf'] and first_iter_fold4_rf == True:
                        TEST_D_prob_rf = test_predict1
                        TEST_D_vote_rf = test_predict
                        first_iter_fold4_rf = False
                    elif clf_to_use == ['rf'] and first_iter_fold4_rf == False:
                        TEST_D_prob_rf = TEST_D_prob_rf + test_predict1
                        TEST_D_vote_rf = np.vstack((TEST_D_vote_rf, test_predict))

                    if clf_to_use == ['svm'] and first_iter_fold4_svm == True:
                        TEST_D_prob_svm = test_predict1
                        TEST_D_vote_svm = test_predict
                        first_iter_fold4_svm = False
                    elif clf_to_use == ['svm'] and first_iter_fold4_svm == False:
                        TEST_D_prob_svm = TEST_D_prob_svm + test_predict1
                        TEST_D_vote_svm = np.vstack((TEST_D_vote_svm, test_predict))
                elif j==5:#fold 5
                    if first_iter_fold5 == True:
                        TEST_E_prob = test_predict1
                        TEST_E_vote = test_predict
                        test_E=Y_test#SAVE true labels for this fold
                        first_iter_fold5 = False
                    else:
                        TEST_E_prob = TEST_E_prob + test_predict1
                        TEST_E_vote = np.vstack((TEST_E_vote, test_predict))
                    if clf_to_use == ['rf'] and first_iter_fold5_rf == True:
                        TEST_E_prob_rf = test_predict1
                        TEST_E_vote_rf = test_predict
                        first_iter_fold5_rf = False
                    elif clf_to_use == ['rf'] and first_iter_fold5_rf == False:
                        TEST_E_prob_rf = TEST_E_prob_rf + test_predict1
                        TEST_E_vote_rf = np.vstack((TEST_E_vote_rf, test_predict))

                    if clf_to_use == ['svm'] and first_iter_fold5_svm == True:
                        TEST_E_prob_svm = test_predict1
                        TEST_E_vote_svm = test_predict
                        first_iter_fold5_svm = False
                    elif clf_to_use == ['svm'] and first_iter_fold5_svm == False:
                        TEST_E_prob_svm = TEST_E_prob_svm + test_predict1
                        TEST_E_vote_svm = np.vstack((TEST_E_vote_svm, test_predict))
                j=j+1 #next fold

        if clf_to_use==['rf']:
            time_rf=time()-start1
        elif clf_to_use==['svm']:
            time_svm=time()-start1


    def five_fold_prob_vote(TEST_vote,TEST_prob,Y_test):
        LABELS=[]

        for i in range(TEST_vote.shape[1]):
            label_0=list(TEST_vote[:, i]).count(0)
            label_1=list(TEST_vote[:, i]).count(1)
            label_2=list(TEST_vote[:, i]).count(2)
            #Majority priority
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


        prec_vote, recall_vote, f_vote, support = precision_recall_fscore_support(Y_test, LABELS, labels=[2, 1, 0],
                                                                   beta=1)
        Conf_vote = confusion_matrix(Y_test, LABELS, labels=[2, 1, 0])
        #print("Confusion matrix ", Conf_vote)
        #print("precision ", prec_vote)
        #print("recall ", recall_vote)
        #print("support ", support)
        accuracy_vote = accuracy_score(Y_test, LABELS)
        #print ("accuracy ", accuracy_vote)
        #print ("f ", f_vote)
        favg_vote = (f_vote[0] + f_vote[1]) * 0.5
        #print ("favg test set : ", favg_vote)



        return favg_vote,0, Conf_vote,0, prec_vote, 0, recall_vote, 0, accuracy_vote, 0, f_vote, 0


    #print("For Fold A.....")
    #print("For All classifiers.........................")
    favg_vote_A, favg_prob_A, Conf_vote_A, Conf_prob_A, prec_vote_A, prec_prob_A, recall_vote_A, recall_prob_A, accuracy_vote_A, accuracy_prob_A, f_vote_A, f_prob_A = five_fold_prob_vote(
        TEST_A_vote, TEST_A_prob, test_A)
    #print("For Random Forest classifier...............")
    fg_v,fg_p,c_v,c_p,p_v,p_p,r_v,r_p,a_v,a_p,f_v,f_p= five_fold_prob_vote(TEST_A_vote_rf, TEST_A_prob_rf, test_A)
    prec_folds_rf=[p_v];rec_folds_rf=[r_v];f_folds_rf=[f_v];acc_folds_rf=[a_v];conf_folds_rf=[c_v];favg_folds_rf=[fg_v]
    #print("For SVM classifier...............")
    fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p= five_fold_prob_vote(TEST_A_vote_svm, TEST_A_prob_svm, test_A)
    prec_folds_svm=[p_v];rec_folds_svm=[r_v];f_folds_svm=[f_v];acc_folds_svm=[a_v];conf_folds_svm=[c_v];favg_folds_svm=[fg_v]

    #print("For RF SVM classifier...............")
    time_rf_svm = time_rf + time_svm
    TEST_A_vote_svm_rf = np.vstack((TEST_A_vote_svm, TEST_A_vote_rf))
    TEST_A_prob_svm_rf = TEST_A_prob_svm+ TEST_A_prob_rf
    fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p= five_fold_prob_vote(TEST_A_vote_svm_rf, TEST_A_prob_svm_rf, test_A)
    prec_folds_rf_svm=[p_v];rec_folds_rf_svm=[r_v];f_folds_svm_rf=[f_v];
    acc_folds_rf_svm=[a_v];conf_folds_svm_rf=[c_v];favg_folds_rf_svm=[fg_v]

    #print("For Fold B.....")
    #print("For All classifiers.........................")
    favg_vote_B, favg_prob_B, Conf_vote_B, Conf_prob_B, prec_vote_B, prec_prob_B, recall_vote_B, recall_prob_B, accuracy_vote_B, accuracy_prob_B, f_vote_B, f_prob_B = five_fold_prob_vote(
        TEST_B_vote, TEST_B_prob, test_B)
    #print("For Random Forest classifier...............")
    fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p= five_fold_prob_vote(TEST_B_vote_rf, TEST_B_prob_rf,
                                                                           test_B)
    prec_folds_rf.append(p_v);rec_folds_rf.append(r_v);f_folds_rf.append(f_v);acc_folds_rf.append(a_v);conf_folds_rf.append(c_v);favg_folds_rf.append(fg_v)
    #print("For SVM classifier...............")
    fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p = five_fold_prob_vote(TEST_B_vote_svm, TEST_B_prob_svm,
                                                                           test_B)
    prec_folds_svm.append(p_v);rec_folds_svm.append(r_v);f_folds_svm.append(f_v);acc_folds_svm.append(a_v);conf_folds_svm.append(c_v);favg_folds_svm.append(fg_v)

    #print("For RF SVM classifier...............")
    TEST_B_prob_svm_rf = TEST_B_prob_svm+ TEST_B_prob_rf
    time_rf_svm = time_rf + time_svm
    TEST_B_vote_svm_rf = np.vstack((TEST_B_vote_svm, TEST_B_vote_rf))
    fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p = five_fold_prob_vote(TEST_B_vote_svm_rf,
                                                                           TEST_B_prob_svm_rf, test_B)
    prec_folds_rf_svm.append(p_v);rec_folds_rf_svm.append(r_v);f_folds_svm_rf.append(f_v);
    acc_folds_rf_svm.append(a_v);conf_folds_svm_rf.append(c_v);favg_folds_rf_svm.append(fg_v)
    if cv>=3:
        #print("For Fold C.....")
        #print("For All classifiers.........................")
        favg_vote_C, favg_prob_C, Conf_vote_C, Conf_prob_C, prec_vote_C, prec_prob_C, recall_vote_C, recall_prob_C, accuracy_vote_C, accuracy_prob_C, f_vote_C, f_prob_C = five_fold_prob_vote(
            TEST_C_vote, TEST_C_prob, test_C)
        #print("For Random Forest classifier...............")
        fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p = five_fold_prob_vote(TEST_C_vote_rf, TEST_C_prob_rf,
                                                                               test_C)
        prec_folds_rf.append(p_v);rec_folds_rf.append(r_v);f_folds_rf.append(f_v);acc_folds_rf.append(a_v);conf_folds_rf.append(c_v);favg_folds_rf.append(fg_v)
        #print("For SVM classifier...............")
        fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p = five_fold_prob_vote(TEST_C_vote_svm, TEST_C_prob_svm,
                                                                               test_C)
        prec_folds_svm.append(p_v);rec_folds_svm.append(r_v);f_folds_svm.append(f_v);acc_folds_svm.append(a_v);conf_folds_svm.append(c_v);favg_folds_svm.append(fg_v)
        #print("For RF SVM classifier...............")
        TEST_C_prob_svm_rf = TEST_C_prob_svm+ TEST_C_prob_rf
        time_rf_svm = time_rf + time_svm
        TEST_C_vote_svm_rf = np.vstack((TEST_C_vote_svm, TEST_C_vote_rf))
        fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p = five_fold_prob_vote(TEST_C_vote_svm_rf,
                                                                               TEST_C_prob_svm_rf, test_C)
        prec_folds_rf_svm.append(p_v);rec_folds_rf_svm.append(r_v);f_folds_svm_rf.append(f_v);
        acc_folds_rf_svm.append(a_v);conf_folds_svm_rf.append(c_v);favg_folds_rf_svm.append(fg_v)

    if cv==5:
        #print("For Fold D.....")
        #print("For All classifiers.........................")
        favg_vote_D, favg_prob_D, Conf_vote_D, Conf_prob_D, prec_vote_D, prec_prob_D, recall_vote_D, recall_prob_D, accuracy_vote_D, accuracy_prob_D, f_vote_D, f_prob_D = five_fold_prob_vote(
            TEST_D_vote, TEST_D_prob, test_D)
        #print("For Random Forest classifier...............")
        fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p = five_fold_prob_vote(TEST_D_vote_rf, TEST_D_prob_rf,
                                                                               test_D)
        prec_folds_rf.append(p_v);
        rec_folds_rf.append(r_v);
        f_folds_rf.append(f_v);
        acc_folds_rf.append(a_v);
        conf_folds_rf.append(c_v);
        favg_folds_rf.append(fg_v)

        #print("For SVM classifier...............")
        fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p = five_fold_prob_vote(TEST_D_vote_svm, TEST_D_prob_svm,
                                                                               test_D)
        prec_folds_svm.append(p_v);
        rec_folds_svm.append(r_v);
        f_folds_svm.append(f_v);
        acc_folds_svm.append(a_v);
        conf_folds_svm.append(c_v);
        favg_folds_svm.append(fg_v)
        #print("For RF SVM classifier...............")
        TEST_D_prob_svm_rf = TEST_D_prob_svm + TEST_D_prob_rf
        time_rf_svm = time_rf + time_svm
        TEST_D_vote_svm_rf = np.vstack((TEST_D_vote_svm, TEST_D_vote_rf))
        fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p = five_fold_prob_vote(TEST_D_vote_svm_rf,
                                                                               TEST_D_prob_svm_rf, test_D)
        prec_folds_rf_svm.append(p_v);
        rec_folds_rf_svm.append(r_v);
        f_folds_svm_rf.append(f_v);
        acc_folds_rf_svm.append(a_v);
        conf_folds_svm_rf.append(c_v);
        favg_folds_rf_svm.append(fg_v)


        #print("For Fold E.....")
        #print("For All classifiers.........................")
        favg_vote_E, favg_prob_E, Conf_vote_E, Conf_prob_E, prec_vote_E, prec_prob_E, recall_vote_E, recall_prob_E, accuracy_vote_E, accuracy_prob_E, f_vote_E, f_prob_E = five_fold_prob_vote(
            TEST_E_vote, TEST_E_prob, test_E)
        #print("For Random Forest classifier...............")
        fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p = five_fold_prob_vote(TEST_E_vote_rf, TEST_E_prob_rf,
                                                                               test_E)
        prec_folds_rf.append(p_v);
        rec_folds_rf.append(r_v);
        f_folds_rf.append(f_v);
        acc_folds_rf.append(a_v);
        conf_folds_rf.append(c_v);
        favg_folds_rf.append(fg_v)
        #print("For SVM classifier...............")
        fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p = five_fold_prob_vote(TEST_E_vote_svm, TEST_E_prob_svm,
                                                                               test_E)
        prec_folds_svm.append(p_v);
        rec_folds_svm.append(r_v);
        f_folds_svm.append(f_v);
        acc_folds_svm.append(a_v);
        conf_folds_svm.append(c_v);
        favg_folds_svm.append(fg_v)

        #print("For RF SVM classifier...............")
        TEST_E_prob_rf_svm = TEST_E_prob_svm + TEST_E_prob_rf
        time_rf_svm = time_rf + time_svm
        TEST_E_vote_svm_rf = np.vstack((TEST_E_vote_svm, TEST_E_vote_rf))
        fg_v, fg_p, c_v, c_p, p_v, p_p, r_v, r_p, a_v, a_p, f_v, f_p = five_fold_prob_vote(TEST_E_vote_svm_rf,
                                                                               TEST_E_prob_rf_svm, test_E)
        prec_folds_rf_svm.append(p_v);
        rec_folds_rf_svm.append(r_v);
        f_folds_svm_rf.append(f_v);
        acc_folds_rf_svm.append(a_v);
        conf_folds_svm_rf.append(c_v);
        favg_folds_rf_svm.append(fg_v)


    #print("The f-score for the folds Hard vote")

    #print(favg_vote_A,favg_vote_B,favg_vote_C,favg_vote_D,favg_vote_E)
    #print("Computing overall avg score across folds using Hard Vote...")
    Fs = np.array([favg_vote_A, favg_vote_B, favg_vote_C, favg_vote_D, favg_vote_E])

    Fs_rf = np.array(favg_folds_rf)
    Fs_svm = np.array(favg_folds_svm)
    Fs_svm_rf = np.array(favg_folds_rf_svm)
    cvscore = np.mean(Fs)
    cvscore_rf=np.mean(Fs_rf);
    cvscore_svm=np.mean(Fs_svm);
    cvscore_svm_rf=np.mean(Fs_svm_rf);

    std=np.std(Fs)
    std_rf=np.std(Fs_rf);
    std_svm=np.std(Fs_svm);
    std_svm_rf=np.std(Fs_svm_rf);

    #print("avg cvscore ", cvscore)
    #print("std cvscore ",std)
    CVSCORE_hard.append(cvscore)
    STD_hard.append(std)
    prec_folds=[prec_vote_A, prec_vote_B, prec_vote_C, prec_vote_D, prec_vote_E]
    rec_folds=[recall_vote_A, recall_vote_B, recall_vote_C, recall_vote_D, recall_vote_E]
    f_folds=[f_vote_A, f_vote_B, f_vote_C, f_vote_D, f_vote_E]
    acc_folds=[accuracy_vote_A, accuracy_vote_B, accuracy_vote_C, accuracy_vote_D, accuracy_vote_E]
    conf_folds=[Conf_vote_A, Conf_vote_B, Conf_vote_C, Conf_vote_D, Conf_vote_E]
    favg_folds=[favg_vote_A, favg_vote_B, favg_vote_C, favg_vote_D, favg_vote_E]


    PREC_hard.append(prec_folds)
    REC_hard.append(rec_folds)
    F_hard.append(f_folds)
    ACC_hard.append(acc_folds)
    CONF_hard.append(conf_folds)
    FAVG_hard.append(favg_folds)

    #dframe_hard_rf.loc[len(dframe_hard_rf)]=[title,cvscore_rf, std_rf,prec_folds_rf,rec_folds_rf,f_folds_rf,acc_folds_rf,conf_folds_rf,favg_folds_rf,time_rf]
    dframe_hard_rf.loc[len(dframe_hard_rf)]=[cvscore_rf, std_rf,prec_folds_rf,rec_folds_rf,f_folds_rf,acc_folds_rf,conf_folds_rf,favg_folds_rf,time_rf]

    if write_separate_excels:
        dframe_hard_rf.to_excel(xlsfile_hard_rf, encoding='utf-8', index=False)
    #dframe_hard_svm.loc[len(dframe_hard_svm)] = [title, cvscore_svm, std_svm, prec_folds_svm, rec_folds_svm,
    #                                           f_folds_svm, acc_folds_svm, conf_folds_svm, favg_folds_svm, time_svm]
    dframe_hard_svm.loc[len(dframe_hard_svm)] = [cvscore_svm, std_svm, prec_folds_svm, rec_folds_svm,
                                               f_folds_svm, acc_folds_svm, conf_folds_svm, favg_folds_svm, time_svm]
    if write_separate_excels:
        dframe_hard_svm.to_excel(xlsfile_hard_svm, encoding='utf-8', index=False)

    #dframe_hard_svm_rf.loc[len(dframe_hard_svm_rf)] = [title, cvscore_svm_rf, std_svm_rf, prec_folds_rf_svm, rec_folds_rf_svm,
    #                                           f_folds_svm_rf, acc_folds_rf_svm, conf_folds_svm_rf, favg_folds_rf_svm, time_rf_svm]
    dframe_hard_svm_rf.loc[len(dframe_hard_svm_rf)] = [cvscore_svm_rf, std_svm_rf, prec_folds_rf_svm,
                                                       rec_folds_rf_svm,
                                                       f_folds_svm_rf, acc_folds_rf_svm, conf_folds_svm_rf,
                                                       favg_folds_rf_svm, time_rf_svm]
    if write_separate_excels:
        dframe_hard_svm_rf.to_excel(xlsfile_hard_svm_rf, encoding='utf-8', index=False)

    ### writing the scores of the three ensembles in an excel file
    dframe_hard_total.loc[len(dframe_hard_total)] = ["SVM",cvscore_svm, std_svm, prec_folds_svm, rec_folds_svm,
                                               f_folds_svm, acc_folds_svm, conf_folds_svm, favg_folds_svm, time_svm]
    dframe_hard_total.loc[len(dframe_hard_total)] = ["RF",cvscore_rf, std_rf, prec_folds_rf, rec_folds_rf,
                                                     f_folds_rf, acc_folds_rf, conf_folds_rf,
                                                     favg_folds_rf, time_rf]
    dframe_hard_total.loc[len(dframe_hard_total)] = ["RF SVM",cvscore_svm_rf, std_svm_rf, prec_folds_rf_svm, rec_folds_rf_svm,
                                                     f_folds_svm_rf, acc_folds_rf_svm, conf_folds_svm_rf,
                                                     favg_folds_rf_svm, time_rf_svm]
    dframe_hard_total.to_excel(xlsfile_hard_total, encoding='utf-8', index=False)


    #print("Conclusion----------------------------------------------------------------")
    #print("Hard vote")
    #print("For the folds :")
    #print("Prec");print(PREC_hard)
    #print("Recalls");print(REC_hard)
    #print("F measrue");print(F_hard)
    #print("Accuracy");print(ACC_hard)
    #print("confusion");print(CONF_hard)
    #print("Favg");print(FAVG_hard)

    #print("Finally........................................................")
    #print("CV scores for Hard vote")
    #print(CVSCORE_hard)
    #print("STD for Hard vote")
    #print(STD_hard)

    return cvscore_svm, cvscore_rf, cvscore_svm_rf

