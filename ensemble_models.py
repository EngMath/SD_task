
import numpy as np
import majority_cv
import majority_test

print("Running Ensemble models for a dataset of your selection: ")
d = input("Enter 1 to select Hillary dataset, 2 for Feminist dataset, 3 for Abortion, 4 for Atheism, 5 for Climate dataset: ")
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

        print("Computing CV scores for the three ensembles ........")
        cvscore_svm, cvscore_rf, cvscore_svm_rf = majority_cv.compute_ensemble_cv(TARGET)
        print("CV scores:")
        print("Ensemble SVM: ", cvscore_svm)
        print("Ensemble RF: ", cvscore_rf)
        print("Ensemble SVM RF: ", cvscore_svm_rf)
        print("Computing Test scores for the three ensembles .........")
        favg_svm, conf_svm, favg_rf, conf_rf, favg_svm_rf, conf_rf_svm = majority_test.compute_ensemble_test(TARGET)
        print("Test scores:")
        print("Ensemble SVM: ", favg_svm)
        print("Ensemble RF: ", favg_rf)
        print("Ensemble SVM RF: ", favg_svm_rf)

    else:
        ok = 0
        d = input("Please enter a valid number to select the dataset. Enter only 1 or 2 or 3 or 4 or 5 : ")

