
import os

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, InputLayer, Conv1D, MaxPooling1D
from biotransformers import BioTransformers
from tensorflow.keras.callbacks import (EarlyStopping,ModelCheckpoint)
from sklearn.metrics import classification_report 
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
from deepchain.models.utils import confusion_matrix_plot, model_evaluation_accuracy, confusion_matrix
import numpy
import logging

import matplotlib.pyplot as plt

import datetime
from utils import parse_args, create_dataset, split_dataset, prepare_dataset, model_input ,  model_evaluation
from imblearn.over_sampling import RandomOverSampler

def build_model(time_steps, input_shape):
    """PAaM-Predictor model
    Combined CNN and Bi-LSTM, to predict the Pathogenic Amino acids mutations
    """
    custom_model = Sequential(name="PathogenicMutation")
    custom_model.add(InputLayer(input_shape=(time_steps, input_shape)))
    custom_model.add(Conv1D(filters=128, kernel_size=1, activation='relu')) # expected input data shape: (batch_size, timesteps, input_shape)
    custom_model.add(MaxPooling1D(pool_size=3, padding="same"))
    custom_model.add(Bidirectional(LSTM(100, return_sequences=True)))
    custom_model.add(Dense(1, activation="sigmoid"))

    return custom_model
if __name__ == "__main__":
    SEED = 45
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.get_logger().setLevel(logging.ERROR)
    # training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 300
    MODEL_CHECKPOINT = "checkpoint/model_0.hdf5"
    SAVED_MODEL_PATH = ("checkpoint/model_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".hdf5")
    BIOTF_MODEL = "protbert_bfd"
    BIOTF_POOLMODE = "mean"
    DEVICE="cuda"
    over = RandomOverSampler()
    args=parse_args()
    if args.dataset=='BRCA1':
          # Run the train with BRCA1 dataset

          # create dataset and compute embeddings
         data = create_dataset(args.input) #data = create_dataset('data/BRCA1_CTerDom.csv')

         Aa_hydro, labels, sequences= prepare_dataset(data)
         bio_trans = BioTransformers(backend=BIOTF_MODEL, device=DEVICE)
         sequences_embeddings = bio_trans.compute_embeddings(
         sequences, pool_mode=(BIOTF_POOLMODE,))[
         BIOTF_POOLMODE]
         sequences_hydro = np.concatenate((sequences_embeddings,Aa_hydro),axis=1)
         x_train, y_train, x_test, y_test, x_val, y_val=split_dataset(sequences_hydro, labels)

         x_train,x_test,x_val= model_input(x_train, x_test, x_val)
         INPUT_SHAPE = x_train.shape[2]
         TIME_STEPS = x_train.shape[0]
        # build model
         model = build_model(TIME_STEPS, INPUT_SHAPE)
         print(model.summary())
     # compile model
         model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[tf.keras.metrics.BinaryAccuracy()],)
         my_callbacks = [
    #EarlyStopping(monitor="val_loss", min_delta=0, patience=40, verbose=1),
         ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filepath=SAVED_MODEL_PATH,
            save_best_only=True,
        )]
    # fit and train the model
         history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(x_val, y_val), callbacks=my_callbacks)
    #evaluate model, plot PR curve/ROC curve in the main folder of the app and compute scores
         model,x_test,y_test= model_evaluation(model,x_test,y_test,args.output_dir)
    if args.dataset=='PTEN':  # Run the train with BRCA1 dataset

    # create dataset and compute embeddings
         data = create_dataset(args.input) #data = create_dataset('data/Pten_dataset.csv')
         Aa_hydro, labels, sequences= prepare_dataset(data)
         bio_trans = BioTransformers(backend=BIOTF_MODEL, device=DEVICE)
         sequences_embeddings = bio_trans.compute_embeddings(
         sequences, pool_mode=(BIOTF_POOLMODE,))[
         BIOTF_POOLMODE]
         sequences_hydro = np.concatenate((sequences_embeddings,Aa_hydro),axis=1)
         x_train, y_train, x_test, y_test, x_val, y_val=split_dataset(sequences_hydro, labels)

         x_train,x_test,x_val= model_input(x_train, x_test, x_val)
         INPUT_SHAPE = x_train.shape[2]
         TIME_STEPS = x_train.shape[0]
        # build model
         model = build_model(TIME_STEPS, INPUT_SHAPE)
         print(model.summary())
        # compile model
         model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[tf.keras.metrics.BinaryAccuracy()],)
         my_callbacks = [
        #EarlyStopping(monitor="val_loss", min_delta=0, patience=40, verbose=1),
         ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filepath=SAVED_MODEL_PATH,
            save_best_only=True,
         )]
         # fit and train the model
         history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(x_val, y_val), callbacks=my_callbacks)
          #evaluate model, plot PR curve/ROC curve in the main folder of the app and compute scores
         model,x_test,y_test= model_evaluation(model,x_test,y_test,args.output_dir)
    

    





