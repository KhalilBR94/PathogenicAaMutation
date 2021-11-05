""""
WARNINGS: if you run the app locally and don't have a GPU you should choose device='cpu'
"""

from typing import List
from biotransformers import BioTransformers
from tensorflow.keras.models import load_model
import numpy as np
import argparse

from  utils import parse_args, create_dataset, prepare_dataset
import logging


""" evaluation script for mutational model"""
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from utils import create_dataset, split_dataset, prepare_dataset, model_input, model_evaluation 

if __name__ == "__main__":

    # fix the seed
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.get_logger().setLevel(logging.ERROR)
    BIOTF_MODEL = "protbert_bfd"
    BIOTF_POOLMODE = "mean"
    DEVICE="cuda"

    # embedding and convolution parameters

    # eval parameters
    BATCH_SIZE = 32
    args=parse_args()
    #params = vars(args)
    print (args)
    if args.dataset=='BRCA1':  # Run the app with BRCA1 dataset
         MODEL_CHECKPOINT = "checkpoint/model_20211102194517.hdf5"  #BRCA1_dataset from train
         data = create_dataset(args.input) #data = create_dataset('data/BRCA1_CTerDom.csv')

         Aa_hydro, labels, sequences= prepare_dataset(data)
        # compute embeddings and Hydrophobicity representation
         bio_trans = BioTransformers(backend=BIOTF_MODEL, device=DEVICE)
         sequences_embeddings = bio_trans.compute_embeddings(
         sequences, pool_mode=(BIOTF_POOLMODE,))[
         BIOTF_POOLMODE]
         sequences_hydro = np.concatenate((sequences_embeddings,Aa_hydro),axis=1)
         x_train, y_train, x_test, y_test, x_val, y_val=split_dataset(sequences_hydro, labels)

         x_train,x_test,x_val= model_input(x_train, x_test, x_val)    
         # Test(1, 19, 1427)
         # Train (1,94,1427)_ Validation(1,19,1427)
         # load model
         model = load_model(MODEL_CHECKPOINT)
         print(model.summary())

         outputs = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
         print("Test scores:", outputs)
         model,x_test,y_test= model_evaluation(model,x_test,y_test,args.output_dir)

    if args.dataset=='PTEN':  #Run the app with PTEN dataset
         MODEL_CHECKPOINT = "checkpoint/model_20211102195022.hdf5"  #PTEN_dataset from train
         data = create_dataset(args.input) 
         Aa_hydro, labels, sequences= prepare_dataset(data)
    
         # compute embeddings and Hydrophobicity representation
         bio_trans = BioTransformers(backend=BIOTF_MODEL, device=DEVICE)
         sequences_embeddings = bio_trans.compute_embeddings(
         sequences, pool_mode=(BIOTF_POOLMODE,))[
         BIOTF_POOLMODE]
         sequences_hydro = np.concatenate((sequences_embeddings,Aa_hydro),axis=1)
         x_train, y_train, x_test, y_test, x_val, y_val=split_dataset(sequences_hydro, labels)

         x_train,x_test,x_val= model_input(x_train, x_test, x_val)    
         # Test(1, 273, 1249)
         # Train (1,2010,1249)_ Validation(1,274,1249)
         # load model
         model = load_model(MODEL_CHECKPOINT)
         print(model.summary())

         outputs = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
         print("Test scores:", outputs)
         model,x_test,y_test= model_evaluation(model,x_test,y_test,args.output_dir)


  
    
