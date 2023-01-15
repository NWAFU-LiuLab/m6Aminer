# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 09:08:06 2022

@author: 80699
"""
from __future__ import division
import math
from functools import reduce
import operator
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from xgboost import plot_importance
def pseiip(seq):
    eiip1=dict(A=0.1260,C=0.1340,G=0.0806,T=0.1335)
    eiip3={}
    mer3={}
    feq3={}
    for n1 in 'ATCG':
        for n2 in 'ATCG':
            for n3 in 'ATCG':
                eiip3[n1+n2+n3]=eiip1[n1]+eiip1[n2]+eiip1[n3]
                mer3[n1+n2+n3]=0
                feq3[n1+n2+n3]=0
    v=[]
    for i in range(len(seq)):
        v.append(eiip1[seq[i]])
    for i in range(len(seq)-2):
        mer3[seq[i:i+3]]+=1
    for elem in mer3:
        feq3[elem]=float(mer3[elem]/(len(seq)-2))
        v.append(eiip3[elem]*feq3[elem])
    return v
def main():
    
    for z in range(1,11):
        f = open(r'ten_flod.txt', 'r')
        index = list(f)
        f.close()
        for i in range(0, len(index)):
            index[i].replace('\n', '')
            index[i] = int(index[i])

        feature_matrix = []
        label_vector = []
        train_samples = open('P_RNA.fasta', 'r')
        i = 0
        b = 0
        for line in train_samples:
            feature_vector = []
            if i % 2 != 0:
                label_vector.append(1)
                sequence = line.strip()
                feature_vector.extend(pseiip(sequence))
                feature_matrix.append(feature_vector)
                b = b + 1
            i = i + 1
            if len(label_vector)==3700:
                break
        train_samples.close()

        train_samples = open('N_RNA.fasta', 'r')
        n = 0
        b = 0
        i = 0
        for line in train_samples:
            feature_vector = []
            if n % 2 != 0:
                if index[i] == z:
                    label_vector.append(0)
                    sequence = line.strip()
                    feature_vector.extend(pseiip(sequence))
                    feature_matrix.append(feature_vector)
                    b = b + 1
                i = i + 1
            n = n + 1
            if len(label_vector) == 7400:
                break
        train_samples.close()
        feature_array = np.array(feature_matrix, dtype=np.float32)

        print(feature_array.shape)

        min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1, 1))
        feature_scaled= min_max_scaler.fit_transform(feature_array)
        X=feature_scaled
        y=label_vector

        clf = XGBClassifier(learning_rate=0.1,
                            n_estimators=100,
                            booster='dart',
                            eta=0.03,
                            max_depth=8,
                            gamma=0.1,
                            subsample=0.7,
                            colsample_bytree=0.45,
                            rate_drop=0.22,
                            skip_drop=0.57,
                            min_child_weight=2,
                            num_round=935)
        clf.fit(X,y)

        feature_matrix_test = []
        label_vector_test = []
        test_samples = open('Ptest_RNA.fasta', 'r')
        i = 0
        b = 0
        for line in test_samples:
            feature_vector_test = []
            if i % 2 != 0:
                label_vector_test.append(1)
                sequence = line.strip()
                feature_vector_test.extend(pseiip(sequence))
                feature_matrix_test.append(feature_vector_test)
                b = b + 1
            i = i + 1
            if len(label_vector_test)==320:
                break
        test_samples.close()

        test_samples = open('Ntest_RNA.fasta', 'r')
        n = 0
        b = 0
        i = 0
        for line in test_samples:
            feature_vector_test = []
            if n % 2 != 0:
                label_vector_test.append(0)
                sequence = line.strip()
                feature_vector_test.extend(pseiip(sequence))
                feature_matrix_test.append(feature_vector_test)
            n = n + 1
            if len(label_vector_test) == 640:
                break

        X_test = np.array(feature_matrix_test, dtype=np.float32)
        X_test = min_max_scaler.fit_transform(X_test)
        y_test = label_vector_test

        print(X_test.shape)
        print(clf.score(X_test,y_test))
        predict_y_test = clf.predict(X_test)
            
        TP=0
        TN=0
        FP=0
        FN=0 
        for i in range(0,len(y_test)):
            if int(y_test[i])==1 and int(predict_y_test[i])==1:
                TP=TP+1
            elif int(y_test[i])==1 and int(predict_y_test[i])==0:
                FN=FN+1
            elif int(y_test[i])==0 and int(predict_y_test[i])==0:
                TN=TN+1
            elif int(y_test[i])==0 and int(predict_y_test[i])==1:
                FP=FP+1
        Sn=float(TP)/(TP+FN)
        Sp=float(TN)/(TN+FP)
        ACC=float((TP+TN))/(TP+TN+FP+FN)
        prob_predict_y_test = clf.predict_proba(X_test)
        predictions_test = prob_predict_y_test[:, 1]

        y_validation=np.array(y_test,dtype=int)
        fpr, tpr, thresholds =metrics.roc_curve(y_validation, predictions_test,pos_label=1)
        roc_auc = auc(fpr, tpr)
            #print('AdaBoostClassifier AUC:%s'%roc_auc)
        F1 = float(2 * TP / (2 * TP + FP + FN))
        mcc = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
        mcc = mcc ** 0.5
        MCC = float((TP * TN - FP * FN) / mcc)

        print('XGB Accuracy:%.3f'%ACC)
        print('XGB AUC:%.3f'%roc_auc)
        print('XGB Sensitive:%.3f'%Sn)
        print('XGB Specificity:%.3f'%Sp)
        print('XGB F1:%.3f'%F1)
        print('XGB MCC:%.3f'%MCC)

if __name__=='__main__':
    main()
