from __future__ import division
import patterns
from igraph import *
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import math
import matplotlib.pyplot as plt
import argparse
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from functools import reduce
from sklearn.metrics import roc_curve, auc
import operator
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from repDNA.psenac import SCPseTNC
import subprocess
import joblib
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def patterns(seq, win):
    """
    Generate k-mers: subsequences of length k
    contained in a biological sequence.
    """
    seqlen = len(seq)
    for i in range(seqlen):
        j = seqlen if i + win > seqlen else i + win
        yield seq[i:j]
        if j == seqlen: break
    return

def hash1(seq):
    binary_dictionary={'A':0, 'C':1, 'G':2, 'T':3}
    seq=seq.strip('N')
    seq_len=len(seq)
    cnt=[]
    for i in seq:
        cnt.append(binary_dictionary[i])
    v2=[]
    for p in range(0,seq_len-1):
        v2.append(4*cnt[p]+cnt[p+1])
    return v2
def FE(seq):
    len_seq=len(seq)
    n={}#平均数
    u={}#期望
    D={}#方差
    f={}#例如，第i个位置时AA的数量
    v=[]
    for n1 in 'ATGC':
        for n2 in 'ATGC':
            n[n1+n2]=0
            f[n1+n2]=0
            u[n1+n2]=0
            D[n1+n2]=0
    for i in range(0,len_seq-1):###采用累加累乘计算二元核苷酸的平均次数、期望及其方差
        n[seq[i:i+2]]+=1
        f[seq[i:i+2]]=1
        u[seq[i:i+2]]+=(i+1)*f[seq[i:i+2]]/float(n[seq[i:i+2]])
        t=(i+1-u[seq[i:i+2]])*(i+1-u[seq[i:i+2]])
        D[seq[i:i+2]]+=t*f[seq[i:i+2]]/float(n[seq[i:i+2]]*(len_seq-1))
    for n1 in 'ATGC':
        for n2 in 'ATGC':
            v.append(n[n1+n2])
            v.append(u[n1+n2])
            v.append(D[n1+n2])
    return v

def ksnpf(seq):
    kn = 5
    freq = []
    v = []
    for i in range(0, kn):
        freq.append({})
        for n1 in 'ATCGN':
            freq[i][n1] = {}
            for n2 in 'ATCGN':
                freq[i][n1][n2] = 0
    seq = seq.strip('N')
    seq_len = len(seq)
    for k in range(0, kn):
        for i in range(seq_len - k - 1):
            n1 = seq[i]
            n2 = seq[i + k + 1]
            freq[k][n1][n2] += 1
    for i in range(0, kn):
        for n1 in 'ATCG':
            for n2 in 'ATCG':
                v.append(freq[i][n1][n2])
    return v


def kmer(seq):
    mer2 = {}
    mer3 = {}
    mer4 = {}
    for n1 in 'ATCG':
        for n2 in 'ATCG':
            mer2[n1 + n2] = 0
            for n3 in 'ATCG':
                mer3[n1 + n2 + n3] = 0
                for n4 in 'ATCG':
                    mer4[n1 + n2 + n3 + n4] = 0
    seq = seq.replace('N', '')
    seq_len = len(seq)
    for p in range(0, seq_len - 3):
        mer2[seq[p:p + 2]] += 1
        mer3[seq[p:p + 3]] += 1
        mer4[seq[p:p + 4]] += 1
    mer2[seq[p + 1:p + 3]] += 1
    mer2[seq[p + 2:p + 4]] += 1
    mer3[seq[p + 1:p + 4]] += 1
    v2 = []
    v3 = []
    v4 = []
    for n1 in 'ACGT':
        for n2 in 'ACGT':
            v2.append(mer2[n1 + n2])
            for n3 in 'ACGT':
                v3.append(mer3[n1 + n2 + n3])
                for n4 in 'ACGT':
                    v4.append(mer4[n1 + n2 + n3 + n4])
    v = v2 + v3 + v4
    return v


def npf(seq):
    binary_dictionary = {'A': [1, 1, 1], 'T': [0, 1, 0], 'G': [1, 0, 0], 'C': [0, 0, 1], 'N': [0, 0, 0]}
    cnt = []
    for i in seq:
        cnt.append(binary_dictionary[i])
    return reduce(operator.add, cnt)


def DBE(seq):
    AA_dict = {
        'AA': [0, 0, 0, 0],
        'AC': [0, 0, 0, 1],
        'AG': [0, 0, 1, 0],
        'AT': [0, 0, 1, 1],
        'CA': [0, 1, 0, 0],
        'CC': [0, 1, 0, 1],
        'CG': [0, 1, 1, 0],
        'CT': [0, 1, 1, 1],
        'GA': [1, 0, 0, 0],
        'GC': [1, 0, 0, 1],
        'GG': [1, 0, 1, 0],
        'GT': [1, 0, 1, 1],
        'TA': [1, 1, 0, 0],
        'TC': [1, 1, 0, 1],
        'TG': [1, 1, 1, 0],
        'TT': [1, 1, 1, 1],
    }

    code = []
    for j in range(len(seq) - 1):
        if seq[j] + seq[j + 1] in AA_dict:
            code += AA_dict[seq[j] + seq[j + 1]]
        else:
            code += [0.5, 0.5, 0.5, 0.5]
    return code


def pseiip(seq):
    eiip1 = dict(A=0.1260, C=0.1340, G=0.0806, T=0.1335)
    eiip3 = {}
    mer3 = {}
    feq3 = {}
    for n1 in 'ATCG':
        for n2 in 'ATCG':
            for n3 in 'ATCG':
                eiip3[n1 + n2 + n3] = eiip1[n1] + eiip1[n2] + eiip1[n3]
                mer3[n1 + n2 + n3] = 0
                feq3[n1 + n2 + n3] = 0
    v = []
    for i in range(len(seq)):
        v.append(eiip1[seq[i]])
    for i in range(len(seq) - 2):
        mer3[seq[i:i + 3]] += 1
    for elem in mer3:
        feq3[elem] = float(mer3[elem] / (len(seq) - 2))
        v.append(eiip3[elem] * feq3[elem])
    return v

def PseKNC_code(seq):
    binary_dictionary = {'A': [1, 1, 1], 'T': [0, 0, 1], 'G': [1, 0, 0], 'C': [0, 1, 0], 'N': [0, 0, 0]}
    nucleic_dictionary = {'A': 0, 'T': 0, 'C': 0, 'G': 0, 'N': 0}
    cnt = []
    p = 0
    for i in seq:
        temp = []
        p = p + 1
        nucleic_dictionary[i] += 1
        temp = list(binary_dictionary[i])
        temp.append(nucleic_dictionary[i] / p)
        cnt.append(temp)
    return reduce(operator.add, cnt)

def main():

    for z in range(1,11):
        f = open(r'ten_flod.txt', 'r')
        index = list(f)
        f.close()
        for i in range(0, len(index)):
            index[i].replace('\n', '')
            index[i] = int(index[i])

        sc_psetnc = SCPseTNC(lamada=2, w=0.05)
        pos_vec3 = sc_psetnc.make_scpsetnc_vec(open('P_RNA.fasta'))
        neg_vec3 = sc_psetnc.make_scpsetnc_vec(open('N_RNA.fasta'))
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
                feature_vector.extend(pseiip(sequence)
                                      + DBE(sequence) +
                                      npf(sequence) +
                                      PseKNC_code(sequence) +
                                      kmer(sequence) +
                                      pos_vec3[b] +
                                      ksnpf(sequence) +
                                      hash1(sequence) +
                                      FE(sequence))
                feature_matrix.append(feature_vector)
                b = b + 1
            i = i + 1
            if len(label_vector) == 3700:
                break
        train_samples.close()

        train_samples = open('./N_RNA.fasta', 'r')
        n = 0
        b = 0
        i = 0
        for line in train_samples:
            feature_vector = []
            if n % 2 != 0:
                if index[i] == z:
                    label_vector.append(0)
                    sequence = line.strip()
                    feature_vector.extend(pseiip(sequence)
                                          + DBE(sequence)
                                          + npf(sequence)
                                          + PseKNC_code(sequence)
                                          + kmer(sequence)
                                          + neg_vec3[i]
                                          + ksnpf(sequence)
                                          + hash1(sequence)
                                          + FE(sequence))
                    feature_matrix.append(feature_vector)
                    b = b + 1
                i = i + 1
            n = n + 1
            if len(label_vector) == 7400:
                break

        feature_array = np.array(feature_matrix, dtype=np.float32)

        f = open(r'feature_importance.txt', 'r')
        rank = list(f)
        f.close()
        for i in range(0, len(rank)):
            rank[i].replace('\n', '')
            rank[i] = int(rank[i])

        max_key = 300


        min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1, 1))
        feature_scaled = min_max_scaler.fit_transform(feature_array)
        X = feature_scaled
        y = label_vector

        print(X.shape)

        X = X[..., rank]
        X = X[..., 0:max_key]

        print(X.shape)

        clf = CatBoostClassifier(iterations=2000,
                             learning_rate=0.03,
                             bootstrap_type='Bayesian',
                             bagging_temperature=1,
                             sampling_frequency='PerTree',
                             leaf_estimation_method='Newton',
                             od_type='IncToDec',
                             task_type='GPU',
                             depth=9,
                             loss_function='MultiClass',
                             random_seed=0)

        clf.fit(X, y)

        pos_vec3_test = sc_psetnc.make_scpsetnc_vec(open('Ptest_RNA.fasta'))
        neg_vec3_test = sc_psetnc.make_scpsetnc_vec(open('Ntest_RNA.fasta'))
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
                feature_vector_test.extend(pseiip(sequence) +
                                           DBE(sequence) +
                                           npf(sequence) +
                                           PseKNC_code(sequence) +
                                           kmer(sequence) +
                                           pos_vec3_test[b] +
                                           ksnpf(sequence) +
                                           hash1(sequence) +
                                           FE(sequence))
                feature_matrix_test.append(feature_vector_test)
                b = b + 1
            i = i + 1
            if len(label_vector_test) == 320:
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
                feature_vector_test.extend(pseiip(sequence) +
                                           DBE(sequence) +
                                           npf(sequence) +
                                           PseKNC_code(sequence) +
                                           kmer(sequence) +
                                           neg_vec3_test[i] +
                                           ksnpf(sequence) +
                                           hash1(sequence) +
                                           FE(sequence))
                feature_matrix_test.append(feature_vector_test)
                i = i + 1
            n = n + 1
            if len(label_vector_test) == 640:
                break
        print(len(label_vector_test))

        X_test = np.array(feature_matrix_test, dtype=np.float32)
        X_test = min_max_scaler.fit_transform(X_test)
        y_test = label_vector_test

        X_test = X_test[..., rank]
        X_test = X_test[..., 0:max_key]
        print(X_test.shape)

        print(clf.score(X_test, y_test))
        predict_y_test = clf.predict(X_test)

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(0, len(y_test)):
            if int(y_test[i]) == 1 and int(predict_y_test[i]) == 1:
                TP = TP + 1
            elif int(y_test[i]) == 1 and int(predict_y_test[i]) == 0:
                FN = FN + 1
            elif int(y_test[i]) == 0 and int(predict_y_test[i]) == 0:
                TN = TN + 1
            elif int(y_test[i]) == 0 and int(predict_y_test[i]) == 1:
                FP = FP + 1
        Sn = float(TP) / (TP + FN)
        Sp = float(TN) / (TN + FP)
        ACC = float((TP + TN)) / (TP + TN + FP + FN)
        prob_predict_y_test = clf.predict_proba(X_test)
        predictions_test = prob_predict_y_test[:, 1]
        print(predictions_test)

        y_validation = np.array(y_test, dtype=int)
        fpr, tpr, thresholds = metrics.roc_curve(y_validation, predictions_test, pos_label=1)
        roc_auc = auc(fpr, tpr)

        F1 = float(2 * TP / (2 * TP + FP + FN))
        mcc = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
        mcc = mcc ** 0.5
        MCC = float((TP * TN - FP * FN) / mcc)

        print('CatBoost Accuracy:%.3f' % ACC)
        print('CatBoost AUC:%.3f' % roc_auc)
        print('CatBoost Sensitive:%.3f' % Sn)
        print('CatBoost Specificity:%.3f' % Sp)
        print('CatBoost F1:%.3f' % F1)
        print('CatBoost MCC:%.3f' % MCC)


if __name__ == '__main__':
    main()
