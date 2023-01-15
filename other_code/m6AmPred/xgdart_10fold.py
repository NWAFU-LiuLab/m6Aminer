# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 08:36:37 2022

@author: 80699
"""

from __future__ import division
import numpy as np
import os
import operator
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_validate
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from functools import reduce
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier

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
    for z in range(1, 11):
        feature_matrix=[]
        label_vector=[]
        train_samples=open('P_RNA.fasta','r')
        i=0
        b=0
        for line in train_samples:
            feature_vector=[]
            if i%2!=0:
                label_vector.append(1)
                sequence=line.strip()
                feature_vector.extend(pseiip(sequence))
                feature_matrix.append(feature_vector)
                b=b+1
            i=i+1
            if len(label_vector) == 3700:
                break
        train_samples.close()

        f = open(r'ten_flod.txt', 'r')
        index = list(f)
        f.close()
        for i in range(0, len(index)):
            index[i].replace('\n', '')
            index[i] = int(index[i])

        train_samples=open('N_RNA.fasta','r')
        n=0
        b=0
        i=0
        for line in train_samples:
            feature_vector=[]
            if n%2!=0:
                if index[i]==z:
                    label_vector.append(0)
                    sequence=line.strip()
                    feature_vector.extend(pseiip(sequence))
                    feature_matrix.append(feature_vector)
                    b=b+1
                i=i+1
            n=n+1
            if len(label_vector)==7400:
                break
        train_samples.close()
        feature_array = np.array(feature_matrix,dtype=np.float32)
        min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1, 1))
        feature_scaled= min_max_scaler.fit_transform(feature_array)
        X=feature_scaled
        y=np.array(label_vector)

        print(X.shape)

        ACC_all = []
        AUC_all = []
        Sn_all = []
        Sp_all = []
        F1_all = []
        MCC_all = []

        skf = StratifiedKFold(n_splits=10, random_state=2, shuffle=True)
        skf.get_n_splits(X, y)

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

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

            clf.fit(X_train,y_train)
            print(clf.score(X_test,y_test))
            predict_y_test = clf.predict(X_test)

            TP=0
            TN=0
            FP=0
            FN=0
            mcc=0
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
            F1=float(2*TP/(2*TP+FP+FN))
            mcc=(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
            mcc=mcc**0.5
            MCC=float((TP*TN-FP*FN)/mcc)
            print('XGB Accuracy:%.3f'%ACC)
            print('XGB AUC:%.3f'%roc_auc)
            print('XGB Sensitive:%.3f'%Sn)
            print('XGB Specificity:%.3f'%Sp)
            print('XGB F1:%.3f'%F1)
            print('XGB MCC:%.3f'%MCC)
            ACC_all.append(ACC)
            AUC_all.append(roc_auc)
            Sn_all.append(Sn)
            Sp_all.append(Sp)
            F1_all.append(F1)
            MCC_all.append(MCC)
        print(ACC_all)
        print(AUC_all)
        print(Sn_all)
        print(Sp_all)
        print(F1_all)
        print(MCC_all)
        print(np.mean(ACC_all))
        print(np.mean(AUC_all))
        print(np.mean(Sn_all))
        print(np.mean(Sp_all))
        print(np.mean(F1_all))
        print(np.mean(MCC_all))


if __name__=='__main__':
    main()
