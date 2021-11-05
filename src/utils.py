#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from deepchain.models.utils import confusion_matrix_plot, model_evaluation_accuracy, confusion_matrix
from sklearn.metrics import precision_recall_curve, classification_report 
from sklearn.metrics import roc_curve, auc, roc_auc_score
import argparse
import os


from pathlib import Path

# In[ ]:

def model_input(x_train, x_test, x_val):

    x_train=np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test=np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    x_val=np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1]))
    return x_train, x_test, x_val



# In[2]:

over = RandomOverSampler()

def create_dataset (file_path):
    data=pd.read_csv(file_path, header=None)
    data.columns = ['Data_{}'.format(int(i)+1) for i in data.columns]
    data.columns = [*data.columns[:-1], 'labels']
    
    return data


# In[3]:

def split_dataset(sequences, labels):
    x_train, x_test, y_train, y_test = train_test_split(sequences, np.array(labels), test_size=0.15, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.176, random_state=2) 
    y_train=y_train.astype(float).reshape(y_train.size,1)
    y_test=y_test.astype(float).reshape(y_test.size,1)
    y_val=y_val.astype(float).reshape(y_val.size,1)
    x_train, y_train = over.fit_resample(x_train, y_train)
    return x_train, y_train, x_val, y_val, x_test, y_test

    
# In[4]:


def prepare_dataset(data,  max_seq_length: int = 1200):
    d = np.array(data)
    sample_len = d.shape[0] # number of protein sequences in the whole dataset
    time_len = d.shape[1]-1 #  length of the protein seq in the whole  dataset
#Hydrophilic coding vocabulary for hydrophobicity representation
    vocab1 = {'A': 1.8, 'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H' :-3.2,'I':4.5,
         'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,'X':-3.5,'*':-100}
#Mutation type Benign:0; Pathogenic:1
    tags = {'0':0, '1':1}
#prepare data splits for embeddings + hydrophbicity index
    data=np.array(data)
    data_len = data.shape[0]
    data_seqlen = data.shape[1] -1

    sequences = data[:, 0:data_seqlen]
    labels = data[:, data_seqlen]
    #labels=labels.astype(float).reshape(labels.size,1)
#replace each Aa of sequences with hydrophobicity index
    Aa_hydro = [[vocab1[word] for word in sentence] for sentence in sequences]
    Aa_hydro = np.array(Aa_hydro).reshape(data_len,data_seqlen)
    labels = labels.reshape(data_len,1)
    return Aa_hydro, labels, sequences


# In[ ]:

def model_evaluation(model,x_test,y_test,output_folder):

    y_pred = model.predict(x_test)   
    prediction=pd.DataFrame(y_pred.flatten()).rename(columns={0:'probability'}) #convert preds to dataframe
    y_pred=prediction['probability']

    prediction['targets']=pd.DataFrame(y_test) #convert labels from numpy array to DataFrame and convert elements from object to float
    y_pred=prediction['probability']
    y_true=prediction['targets']
    #Plot Precision-Recall
    # calculate model precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure()
    plt.plot(precision, recall, color='green', label='protbert_bfd_CNN_BiLSTM')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve ')
    plt.legend()
    plt.show()
    plt.savefig(output_folder+'/PR_curve.png')
    #plot roc_auc curve
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    #roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    plt.savefig(output_folder+'/ROC_curve.png')
    auc = roc_auc_score(y_true, y_pred)
    c=print('AUC Score: %.2f' % auc)
    # Plot confusion matrix
    model_evaluation_accuracy(y_true, y_pred)
    confusion_matrix_plot(y_true, (y_pred> 0.5).astype(int), ["0", "1"])
    matrix_index = ["0","1"]
    y_pred= np.around(y_pred)
    outputs = model.evaluate(
    x_test, y_test
    )
    b=print("Test Loss:", outputs)
    a=print(classification_report(y_true, y_pred,target_names=matrix_index))
    return a,b,c
def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Arguments to run the application")
    parser.add_argument("-d","--dataset", type=str, help="[Required] Choose PTEN or BRCA1 dataset.", required=True)
    parser.add_argument("-i",
        "--input",
        type=str,
        help="[Required] File path for PTEN or BRCA1 dataset.", required=True
    )
    parser.add_argument("-o",
        "--output_dir",
        type=str,
        default="src/Results",
        help="[Optional] Folder for output results (ROC_curve and PR_curve)."
    )
    args = parser.parse_args()

    return args
