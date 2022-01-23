
import numpy as np
import majority_cv
import majority_test


TARGET = "Hillary Clinton"
print("Running Ensemble models for Hillary Clinton dataset...")
print(" Computing CV scores for the three ensembles for Hillary dataset ........")
cvscore_svm, cvscore_rf, cvscore_svm_rf=majority_cv.compute_ensemble_cv(TARGET)
print(" Computing Test scores for the three ensembles for Hillary dataset.........")
favg_svm, hc_conf_svm, favg_rf, hc_conf_rf, favg_svm_rf, hc_conf_svm_rf=majority_test.compute_ensemble_test(TARGET)
print(" CV scores for Hillary target:")
print("  Ensemble SVM: ", cvscore_svm)
print("  Ensemble RF: ", cvscore_rf)
print("  Ensemble SVM RF: ", cvscore_svm_rf)
print(" Test scores for Hillary target:")
print("  Ensemble SVM: ", favg_svm)
print("  Ensemble RF: ", favg_rf)
print("  Ensemble SVM RF: ", favg_svm_rf)

TARGET = "Feminist Movement"
print("Running Ensemble models for Feminist Movement dataset...")
print(" Computing CV scores for the three ensembles for Feminist dataset........")
cvscore_svm, cvscore_rf, cvscore_svm_rf=majority_cv.compute_ensemble_cv(TARGET)
print(" Computing Test scores for the three ensembles for Feminist dataset.........")
favg_svm, fem_conf_svm, favg_rf, fem_conf_rf, favg_svm_rf, fem_conf_svm_rf=majority_test.compute_ensemble_test(TARGET)
print(" CV scores for Feminist target:")
print("  Ensemble SVM: ", cvscore_svm)
print("  Ensemble RF: ", cvscore_rf)
print("  Ensemble SVM RF: ", cvscore_svm_rf)
print(" Test scores for Feminist target:")
print("  Ensemble SVM: ", favg_svm)
print("  Ensemble RF: ", favg_rf)
print("  Ensemble SVM RF: ", favg_svm_rf)

TARGET = "Legalization of Abortion"
print("Running Ensemble models for Legalization of Abortion dataset...")
print(" Computing CV scores for the three ensembles for Abortion dataset........")
cvscore_svm, cvscore_rf, cvscore_svm_rf=majority_cv.compute_ensemble_cv(TARGET)
print(" Computing Test scores for the three ensembles for Abortion dataset.........")
favg_svm, ab_conf_svm, favg_rf, ab_conf_rf, favg_svm_rf, ab_conf_svm_rf=majority_test.compute_ensemble_test(TARGET)
print(" CV scores for Abortion target:")
print("  Ensemble SVM: ", cvscore_svm)
print("  Ensemble RF: ", cvscore_rf)
print("  Ensemble SVM RF: ", cvscore_svm_rf)
print(" Test scores for Abortion target:")
print("  Ensemble SVM: ", favg_svm)
print("  Ensemble RF: ", favg_rf)
print("  Ensemble SVM RF: ", favg_svm_rf)

TARGET = "Atheism"
print("Running Ensemble models for Atheism dataset....")
print(" Computing CV scores for the three ensembles for Atheism dataset ........")
cvscore_svm, cvscore_rf, cvscore_svm_rf=majority_cv.compute_ensemble_cv(TARGET)
print(" Computing Test scores for the three ensembles for Atheism dataset.........")
favg_svm, ath_conf_svm, favg_rf, ath_conf_rf, favg_svm_rf, ath_conf_svm_rf=majority_test.compute_ensemble_test(TARGET)
print(" CV scores for Atheism target:")
print("  Ensemble SVM: ", cvscore_svm)
print("  Ensemble RF: ", cvscore_rf)
print("  Ensemble SVM RF: ", cvscore_svm_rf)
print(" Test scores for Atheism target:")
print("  Ensemble SVM: ", favg_svm)
print("  Ensemble RF: ", favg_rf)
print("  Ensemble SVM RF: ", favg_svm_rf)

TARGET = "Climate Change is a Real Concern"
print("Running Ensemble models for Climate Change is a Real Concern dataset...")
print(" Computing CV scores for the three ensembles for Climate dataset........")
cvscore_svm, cvscore_rf, cvscore_svm_rf=majority_cv.compute_ensemble_cv(TARGET)
print(" Computing Test scores for the three ensembles for Climate dataset.........")
favg_svm, clm_conf_svm, favg_rf, clm_conf_rf, favg_svm_rf, clm_conf_svm_rf=majority_test.compute_ensemble_test(TARGET)
print(" CV scores for Climate target:")
print("  Ensemble SVM: ", cvscore_svm)
print("  Ensemble RF: ", cvscore_rf)
print("  Ensemble SVM RF: ", cvscore_svm_rf)
print(" Test scores for Climate target:")
print("  Ensemble SVM: ", favg_svm)
print("  Ensemble RF: ", favg_rf)
print("  Ensemble SVM RF: ", favg_svm_rf)


def compute_avg_score(clm_conf,ab_conf,ath_conf,fem_conf,hc_conf):
    def confusion(Conf):
        against_tn = Conf[2][2] + Conf[1][1]
        against_tp = Conf[0][0]
        against_fp = Conf[1][0] + Conf[2][0]
        against_fn = Conf[0][1] + Conf[0][2]
        favor_tp = Conf[1][1]
        favor_fp = Conf[0][1] + Conf[2][1]
        favor_tn = Conf[0][0] + Conf[2][2]
        favor_fn = Conf[1][0] + Conf[1][2]
        return against_tn, against_tp, against_fn, against_fp, favor_tn, favor_tp, favor_fn, favor_fp

    AG_TN=[];AG_TP=[];AG_FN=[]
    AG_FP=[];FV_TN=[];FV_TP=[]
    FV_FN=[];FV_FP=[]

    for Conf in [clm_conf,ab_conf,ath_conf,fem_conf,hc_conf]:
        against_tn, against_tp, against_fn, against_fp, favor_tn, favor_tp, favor_fn, favor_fp=confusion(Conf)
        AG_TN.append(against_tn)
        AG_TP.append(against_tp)
        AG_FN.append(against_fn)
        AG_FP.append(against_fp)
        FV_TN.append(favor_tn)
        FV_TP.append(favor_tp)
        FV_FN.append(favor_fn)
        FV_FP.append(favor_fp)

    AG_TN = np.array(AG_TN)
    AG_TP = np.array(AG_TP)
    AG_FN = np.array(AG_FN)
    AG_FP = np.array(AG_FP)
    FV_TN = np.array(FV_TN)
    FV_TP = np.array(FV_TP)
    FV_FN = np.array(FV_FN)
    FV_FP = np.array(FV_FP)

    AG_prec = float(np.sum(AG_TP)) / (np.sum(AG_TP) + np.sum(AG_FP))
    AG_rec = float(np.sum(AG_TP)) / (np.sum(AG_TP) + np.sum(AG_FN))
    AG_FAVG =2.0*(AG_prec * AG_rec)/(AG_prec + AG_rec)
    FV_prec = float(np.sum(FV_TP)) / (np.sum(FV_TP) + np.sum(FV_FP))
    FV_rec = float(np.sum(FV_TP)) / (np.sum(FV_TP) + np.sum(FV_FN))
    FV_FAVG = 2.0*(FV_prec * FV_rec)/ (FV_prec + FV_rec)

    Total_favg = (FV_FAVG + AG_FAVG) * 0.5
    #print("FV_FAVG ",FV_FAVG)
    #print("AG_FAVG ",AG_FAVG)
    #print("Total_favg ",Total_favg)
    return FV_FAVG,AG_FAVG, Total_favg

F_favor_svm,F_against_svm, Favg_svm=compute_avg_score(clm_conf_svm,ab_conf_svm,ath_conf_svm,fem_conf_svm,hc_conf_svm)
print("Computing Total Test Fscores across the five targets using the three ensembles... ")
print("For Ensemble SVM: ")
print("  F_favor: ",F_favor_svm)
print("  F_against: ",F_against_svm)
print("  Favg: ",Favg_svm )
F_favor_rf,F_against_rf, Favg_rf=compute_avg_score(clm_conf_rf,ab_conf_rf,ath_conf_rf,fem_conf_rf,hc_conf_rf)
print("For Ensemble RF: ")
print("  F_favor: ",F_favor_rf)
print("  F_against: ",F_against_rf)
print("  Favg: ",Favg_rf )
F_favor_svm_rf,F_against_svm_rf, Favg_svm_rf= compute_avg_score(clm_conf_svm_rf,ab_conf_svm_rf,ath_conf_svm_rf,fem_conf_svm_rf,hc_conf_svm_rf)
print("For Ensemble SVM RF: ")
print("  F_favor: ",F_favor_svm_rf)
print("  F_against: ",F_against_svm_rf)
print("  Favg: ",Favg_svm_rf )
print("DONE.")