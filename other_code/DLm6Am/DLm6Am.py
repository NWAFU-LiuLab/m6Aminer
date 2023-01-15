import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.metrics import roc_curve
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torchvision import transforms
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, confusion_matrix, accuracy_score
import random
import argparse
import os
from Bio import SeqIO

def seed_torch(seed=104):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch()
torch.manual_seed(104)

def AA_ONE_HOT(seq):
    one_hot_dict = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }
    coding_arr = np.zeros((len(seq),4), dtype=float)
    for m in range(len(seq)):
        coding_arr[m] = one_hot_dict[seq[m]]
    return coding_arr

def chemical_properties(seq):
    one_hot_dict = {
        'A': [1, 1, 1],
        'C': [0, 0, 1],
        'G': [1, 0, 0],
        'T': [0, 1, 0]
    }
    coding_arr = np.zeros((len(seq), 4), dtype=float)
    # for m in range(len(AA)):
    A_num = 0
    C_num = 0
    G_num = 0
    U_num = 0
    All_num = 0
    for x in seq:
        if x == "A":
            All_num += 1
            A_num += 1
            Density = A_num/All_num
            coding_arr[All_num-1] = one_hot_dict[seq[All_num-1]] + [Density]
        if x == "C":
            All_num += 1
            C_num += 1
            Density = C_num/All_num
            coding_arr[All_num-1] = one_hot_dict[seq[All_num-1]] + [Density]
        if x == "G":
            All_num += 1
            G_num += 1
            Density = G_num/All_num
            coding_arr[All_num-1] = one_hot_dict[seq[All_num-1]] + [Density]
        if x == "T":
            All_num += 1
            U_num += 1
            Density = U_num/All_num
            coding_arr[All_num-1] = one_hot_dict[seq[All_num-1]] + [Density]
    return coding_arr

class newModel1(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=8, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.lstm = nn.LSTM(input_size=8, hidden_size=32, num_layers=2, bidirectional=True, batch_first=True)
        self.lstm_fc = nn.Linear(2624, 256)
        # self.lstm_fc = nn.Linear(2624, 1056)
        # nn.Flatten(),

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 4), stride=(1, 1), padding=0),  # 32*36
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=0),  # 32*33
            nn.MaxPool2d(kernel_size=2),  # 32*17

        )

        self.block3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1344, 128),
            # nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(128, 12),
            nn.LeakyReLU(),
            nn.Linear(12, 2),
            nn.Softmax(dim=1)

        )

    def forward(self, x):
        x1 = self.transformer_encoder(x)
        x1, (hn, hc) = self.lstm(x1, None)
        x2 = nn.Flatten()(x1)
        x3 = self.lstm_fc(x2)
        x3 = nn.functional.relu(x3)
        # print(x2.shape, "x2.shape")
        # print(x.shape, "x.shape")
        c1 = self.transformer_encoder(x)
        c1 = torch.unsqueeze(c1, 1)
        c2 = self.block2(c1)
        c3 = nn.Flatten()(c2)
        # print(c4.shape, "c4.shape")
        all1 = torch.cat([c3, x3], dim=1)
        # print(all1.shape, "all1.shape")
        # end = self.block3(all1)
        # print(end.shape, "end.shape")
        # print(c3.size())
        end = self.block3(all1)
        return end

class newModel2(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=8, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.lstm = nn.LSTM(input_size=8, hidden_size=32, num_layers=2, bidirectional=True, batch_first=True)
        self.lstm_fc = nn.Linear(2624, 256)
        # self.lstm_fc = nn.Linear(2624, 1056)
        # nn.Flatten(),

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 4), stride=(1, 1), padding=0),  # 32*36
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=0),  # 32*33
            nn.MaxPool2d(kernel_size=2),  # 32*17

        )

        self.block3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1344, 128),
            # nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(128, 12),
            nn.LeakyReLU(),
            nn.Linear(12, 2),
            nn.Softmax(dim=1)

        )

    def forward(self, x):
        x1 = self.transformer_encoder(x)
        x1, (hn, hc) = self.lstm(x1, None)
        x2 = nn.Flatten()(x1)
        x3 = self.lstm_fc(x2)
        x3 = nn.functional.relu(x3)
        # print(x2.shape, "x2.shape")
        # print(x.shape, "x.shape")
        c1 = self.transformer_encoder(x)
        c1 = torch.unsqueeze(c1, 1)
        c2 = self.block2(c1)
        c3 = nn.Flatten()(c2)
        # print(c4.shape, "c4.shape")
        all1 = torch.cat([c3, x3], dim=1)
        # print(all1.shape, "all1.shape")
        # end = self.block3(all1)
        # print(end.shape, "end.shape")
        # print(c3.size())
        end = self.block3(all1)
        return end


class newModel3(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=8, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.lstm = nn.LSTM(input_size=8, hidden_size=32, num_layers=2, bidirectional=True, batch_first=True)
        self.lstm_fc = nn.Linear(2624, 256)
        # self.lstm_fc = nn.Linear(2624, 1056)
        # nn.Flatten()

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 4), stride=(1, 1), padding=0),  # 32*36
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(4, 4), stride=(1, 1), padding=0),  # 32*33
            nn.MaxPool2d(kernel_size=2),  # 32*17
            )

        self.block3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2432, 256),
            # nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(256, 24),
            nn.LeakyReLU(),
            nn.Linear(24, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.transformer_encoder(x)
        x1, (hn, hc) = self.lstm(x1, None)
        x2 = nn.Flatten()(x1)
        x3 = self.lstm_fc(x2)
        x3 = nn.functional.relu(x3)
        # print(x2.shape, "x2.shape")
        # print(x.shape, "x.shape")
        c1 = self.transformer_encoder(x)
        c1 = torch.unsqueeze(c1, 1)
        c2 = self.block2(c1)
        c3 = nn.Flatten()(c2)
        # print(c4.shape, "c4.shape")
        all1 = torch.cat([c3, x3], dim=1)
        # print(all1.shape, "all1.shape")
        # end = self.block3(all1)
        # print(end.shape, "end.shape")
        # print(c3.size())
        end = self.block3(all1)
        return end

def main():

    for z in range(1,11):
        f = open(r'ten_flod.txt', 'r')
        index = list(f)
        f.close()
        for i in range(0, len(index)):
            index[i].replace('\n', '')
            index[i] = int(index[i])

        num = 0
        for i in range(len(index)):
            if index[i]==z:
                num +=1

        if num>3700:
            num = 3700
        else:
            num = num

        label_vector = []
        train_samples = open('P_RNA.fasta', 'r')
        i = 0
        b = 0
        feature_vector_one_hot = np.zeros(((3700+num), 41, 4))
        feature_vector_chemical = np.zeros(((3700+num), 41, 4))

        for line in train_samples:
            if i % 2 != 0:
                label_vector.append(1)
                sequence = line.strip()
                feature_vector_one_hot[b] = AA_ONE_HOT(sequence)
                feature_vector_chemical[b] = chemical_properties(sequence)
                b = b + 1
            i = i + 1
            if len(label_vector)==3700:
                break
        train_samples.close()

        train_samples = open('N_RNA.fasta', 'r')
        n = 0
        i = 0
        for line in train_samples:
            if n % 2 != 0:
                if index[i] == z:
                    label_vector.append(0)
                    sequence = line.strip()
                    feature_vector_one_hot[b] = AA_ONE_HOT(sequence)
                    feature_vector_chemical[b] = chemical_properties(sequence)
                    b = b + 1
                i = i + 1
            n = n+1
            if len(label_vector) == 7400:
                break

        train_samples.close()

        feature = np.concatenate((feature_vector_chemical, feature_vector_one_hot), axis=2)

        print(feature.shape)
        print(len(label_vector))
        x_train, x_test, y_train, y_test = train_test_split(feature, label_vector, test_size=0.2, random_state=0)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_label = torch.tensor(y_train).to(torch.long).to(device)
        test_label = torch.tensor(y_test).to(torch.long).to(device)

        train = torch.tensor(x_train).to(torch.float32).to(device)
        test = torch.tensor(x_test).to(torch.float32).to(device)

        trainset = Data.TensorDataset(train, train_label)
        testset = Data.TensorDataset(test, test_label)

        batch_size = 32
        train_iter = torch.utils.data.DataLoader(dataset=trainset,
                                                       shuffle=True,
                                                       batch_size=batch_size)

        test_iter = torch.utils.data.DataLoader(dataset=testset,
                                                      shuffle=True,
                                                      batch_size=batch_size)

        net1 = newModel1().to(device)
        lr1 = 0.00007
        optimizer1 = torch.optim.Adam(net1.parameters(), lr=lr1)
        criterion_model1 = nn.CrossEntropyLoss()

        net2 = newModel2().to(device)
        lr2 = 0.000065
        optimizer2 = torch.optim.Adam(net2.parameters(), lr=lr2)
        criterion_model2 = nn.CrossEntropyLoss()

        net3 = newModel3().to(device)
        lr3 = 0.0002
        optimizer3 = torch.optim.Adam(net3.parameters(), lr=lr3)
        criterion_model3 = nn.CrossEntropyLoss()

        num_epochs = 60
        max_acc = 0
        acc_list = []
        TN1, FP1, FN1, TP1, AUC1 = [], [], [], [], []
        m_acc = []
        labels_all = []
        probs_all = []

        for epoch in range(num_epochs):
            y_predict1 = []
            y_test_class1 = []
            y_predict2 = []
            y_test_class2 = []
            y_predict3 = []
            y_test_class3 = []
            print(f'epoch {epoch + 1}')

            net1.train()
            for seq, label in train_iter:
                pred1 = net1(seq)

                x_p1 = pred1.data.cpu().numpy().tolist()
                y_p1 = label.data.cpu().numpy().tolist()
                for p_list1 in x_p1:
                    y_predict1.append(p_list1)
                y_test_class1 += y_p1

                loss1 = criterion_model1(pred1, label)
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()

            net2.train()
            for seq, label in train_iter:
                pred2 = net2(seq)
                x_p2 = pred2.data.cpu().numpy().tolist()
                y_p2 = label.data.cpu().numpy().tolist()
                for p_list2 in x_p2:
                    y_predict2.append(p_list2)
                y_test_class2 += y_p2

                loss2 = criterion_model2(pred2, label)
                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()

            net3.train()
            for seq, label in train_iter:
                pred3 = net3(seq)
                x_p3 = pred3.data.cpu().numpy().tolist()
                y_p3 = label.data.cpu().numpy().tolist()
                for p_list3 in x_p3:
                    y_predict3.append(p_list3)
                y_test_class3 += y_p3

                loss3 = criterion_model3(pred3, label)
                optimizer3.zero_grad()
                loss3.backward()
                optimizer3.step()

            y_predict1 = np.array(y_predict1)
            y_predict_class1 = np.argmax(y_predict1, axis=1)
            y_test_class1 = np.array(y_test_class1)
            acc1 = accuracy_score(y_predict_class1, y_test_class1)

            y_predict2 = np.array(y_predict2)
            y_predict_class2 = np.argmax(y_predict2, axis=1)
            y_test_class2 = np.array(y_test_class2)
            acc2 = accuracy_score(y_predict_class2, y_test_class2)

            y_predict3 = np.array(y_predict3)
            y_predict_class3 = np.argmax(y_predict3, axis=1)
            y_test_class3 = np.array(y_test_class3)
            acc3 = accuracy_score(y_predict_class3, y_test_class3)

            net1.eval()
            net2.eval()
            net3.eval()

            y_predict_1 = []
            y_predict_2 = []
            y_predict_3 = []
            y_test_class = []

            with torch.no_grad():
                for seq1, label1 in test_iter:
                    pred_1 = net1(seq1)
                    pred_2 = net2(seq1)
                    pred_3 = net3(seq1)

                    x_p1 = pred_1.data.cpu().numpy().tolist()
                    x_p2 = pred_2.data.cpu().numpy().tolist()
                    x_p3 = pred_3.data.cpu().numpy().tolist()
                    y_p = label1.data.cpu().numpy().tolist()

                    for p_list1 in x_p1:
                        y_predict_1.append(p_list1)
                    for p_list2 in x_p2:
                        y_predict_2.append(p_list2)
                    for p_list3 in x_p3:
                        y_predict_3.append(p_list3)

                    y_test_class += y_p

                y_predict_1 = np.array(y_predict_1)
                y_predict_2 = np.array(y_predict_2)
                y_predict_3 = np.array(y_predict_3)
                y_test_class = np.array(y_test_class)

                probs_end1 = ((y_predict_1[:, 1] + y_predict_2[:, 1] + y_predict_3[:, 1]) / 3).flatten().tolist()

                y_predict_class_1 = np.argmax(y_predict_1, axis=1)
                y_predict_class_2 = np.argmax(y_predict_2, axis=1)
                y_predict_class_3 = np.argmax(y_predict_3, axis=1)
                prediction_labels_end = []

                for num in range(len(y_test_class)):
                    judge = y_predict_class_1[num] + y_predict_class_2[num] + y_predict_class_3[num]
                    if judge > 1.5:
                        prediction_labels_end += [1]
                    else:
                        prediction_labels_end += [0]
                prediction_labels_end1 = np.array(prediction_labels_end)
                tn, fp, fn, tp = confusion_matrix(y_test_class, prediction_labels_end1).ravel()
                fpr, tpr, thresholds = roc_curve(y_test_class, probs_end1, pos_label=1)
                roc_auc = auc(fpr, tpr)

                test_acc = accuracy_score(prediction_labels_end1, y_test_class)

                if test_acc > max_acc:
                    max_acc = test_acc
                    probs_end = ((y_predict_1[:, 1] + y_predict_2[:, 1] + y_predict_3[:, 1]) / 3).flatten().tolist()
                    labels = y_test_class.flatten().tolist()
                    epoch1 = epoch
                    TN, FP, FN, TP = tn, fp, fn, tp

            m_acc.append([max_acc])
            labels_all += labels
            probs_all += probs_end

            TN1.append(TN)
            FP1.append(FP)
            FN1.append(FN)
            TP1.append(TP)

        TN = np.sum(TN1, dtype=np.float64)
        FP= np.sum(FP1, dtype=np.float64)
        FN = np.sum(FN1, dtype=np.float64)
        TP = np.sum(TP1, dtype=np.float64)
        fpr1, tpr1, thresholds = roc_curve(labels_all, probs_all)

        roc_auc1 = auc(fpr1, tpr1)  # auc为Roc曲线下的面积

        Sn = float(TP) / (TP + FN)
        Sp = float(TN) / (TN + FP)
        ACC = float((TP + TN)) / (TP + TN + FP + FN)
        F1 = float(2 * TP / (2 * TP + FP + FN))
        mcc = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
        mcc = mcc ** 0.5
        MCC = float((TP * TN - FP * FN) / mcc)

        print('DLm6Am Accuracy:%.3f' % ACC)
        print('DLm6Am AUC:%.3f' % roc_auc1)
        print('DLm6Am Sensitive:%.3f' % Sn)
        print('DLm6Am Specificity:%.3f' % Sp)
        print('DLm6Am F1:%.3f' % F1)
        print('DLm6Am MCC:%.3f' % MCC)


if __name__=='__main__':
    main()
