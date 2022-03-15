# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 07:36:50 2022

@author: Paul Vincent Nonat
"""


import numpy as np
import random
import pandas as pd

import matplotlib.pyplot as plt
import os
import argparse
from sklearn.model_selection import train_test_split 
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  EarlyStopping
from tensorflow.keras import regularizers

from sklearn.preprocessing import StandardScaler


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,matthews_corrcoef

from imblearn.under_sampling import RandomUnderSampler
from collections import Counter



def generate_dataset(random_state):
    transactions = pd.read_csv("../data/transactions_train.csv")
    X = transactions[["type", "amount", "oldbalanceOrig", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]]
    Y = transactions["isFraud"]
    X_train, X_test_val, Y_train, Y_test_val = train_test_split(X, Y, test_size=0.2)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test_val, Y_test_val, test_size=0.5)
    
    print("y_train Fraud: {:.2f}%".format(Y_train.value_counts()[0]/len(Y_train)*100))
    print("y_train Non-fraud: {:.2f}%".format(Y_train.value_counts()[1]/len(Y_train)*100))    
    print("y_test Fraud: {:.2f}%".format(Y_test.value_counts()[0]/len(Y_test)*100))
    print("y_test Non-fraud: {:.2f}%".format(Y_test.value_counts()[1]/len(Y_test)*100))
    nb_classes = len(np.unique(Y_train))
    
    return X_train, Y_train,X_val,Y_val, X_test ,Y_test,nb_classes


def create_model(n_of_features,dropout,l2,lr,random_state,nb_classes):
    print("Creating Model")
    model = Sequential()
    model.add(Dense(units=256, input_dim=n_of_features, kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout, seed=random_state))
    model.add(Dense(units=256, input_dim=n_of_features, kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout, seed=random_state))
    model.add(Dense(units=256, input_dim=n_of_features, kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout, seed=random_state))
    model.add(Dense(units=256, kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout, seed=random_state))
    model.add(Dense(units=128, kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout, seed=random_state))
    model.add(Dense(units=128, kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
#    model.add(Dense(units=nb_classes))
#    model.add(Activation('softmax'))
    model.add(Dense(units=1))
    model.add(Activation('relu'))
#    model.compile(loss='mean_absolute_error',optimizer=Adam(lr=lr))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),optimizer=Adam(lr=lr))    
    model.summary()
    print("Done creating Model")
    return model;

def train(X_train, Y_train,X_val,Y_val, X_test ,Y_test,nb_classes,epochs,batch_size,patience):
    results = {
    "Accuracy": [],
    "F1-Score": [],
    "MCC": [],
    "TP": [],
    "FP": [],
    "FN": [],
    "TN": [], }
    
    print("Distribution of y_train set BEFORE balancing: {}", Counter(Y_train))
    under = RandomUnderSampler(sampling_strategy=0.002)
    X_train, Y_train = under.fit_resample(X_train, Y_train)
 
    print("Distribution of y_train set AFTER balancing: {}", Counter(Y_train))

    #encode categorical inputs  
    type_encoder = LabelEncoder()
    type_names = X_train["type"].unique()
    type_encoder.fit(type_names)
    type_encoder.classes_ = np.append(type_encoder.classes_, "<unknown>")
    
    X_train["type"] = type_encoder.transform(X_train["type"])
    X_val["type"] = type_encoder.transform(X_val["type"])
 
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
#    Y_train = to_categorical(Y_train, nb_classes)
#    Y_test = to_categorical(Y_test, nb_classes)
#    Y_test = to_categorical(Y_test, nb_classes)
    
    #to float32 for tensorflow
    X_train=np.asarray(X_train).astype(np.float32)
    X_val=np.asarray(X_val).astype(np.float32)

    print("Start Training")
    cb =[EarlyStopping(monitor='val_loss', patience=patience, verbose =1, restore_best_weights=True)]    
    history = model.fit(X_train, Y_train,validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1, callbacks= cb)
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(trial_name+' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(trial_name+'_training.png')
    print("training_complete")

    predictions = model.predict(X_val)
    f1score = f1_score(Y_val, predictions)
    accuracy = accuracy_score(Y_val, predictions)
    tn, fp, fn, tp = confusion_matrix(Y_val, predictions).ravel()
    mcc=matthews_corrcoef(Y_val, predictions)

    results["Accuracy"].append(accuracy)
    results["F1-Score"].append(mcc)
    results["MCC"].append(f1score)
    results["TP"].append(tp)
    results["FP"].append(fp)
    results["FN"].append(fn)
    results["TN"].append(tn)
          
        
    results_df = pd.DataFrame(results)
    results_df
    
    #encode categorical inputs  for test set and scale it
    type_encoder = LabelEncoder()
    type_names = X_test["type"].unique()
    type_encoder.fit(type_names)
    type_encoder.classes_ = np.append(type_encoder.classes_, "<unknown>")
    X_test["type"] = type_encoder.transform(X_test["type"])        
    
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    X_test=np.asarray(X_test).astype(np.float32)  
        

    #model test
    
    predictions = model.predict(X_test)
    
    f1score = f1_score(Y_test, predictions)
    accuracy = accuracy_score(Y_test, predictions)
    tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()
    mcc=matthews_corrcoef(Y_test, predictions)

    print("F1Score: {}".format(f1score))
    print("Accuracy: {}".format(accuracy))
    print("MCC: {}".format(mcc))
    
    # serialize model to JSON
    model_json = model.to_json()
    
    
    with open("../MLP_saved_models/"+trial_name+".json", "w") as json_file:
        json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights("../MLP_saved_models/"+trial_name+".h5")


if __name__ == '__main__':

    config = tf.compat.v1.ConfigProto( device_count = {'GPU': 0 } ) 
    sess = tf.compat.v1.Session(config=config) 
    tf.compat.v1.keras.backend.set_session(sess)
    tf.debugging.set_log_device_placement(True)    
    trial_name="MLP"
    epochs=50
    patience=10
    dropout = 0.15
    l2 = 0.00
    lr = 0.0005
    batch_size= 512

    random_state = 42
    os.environ['PYTHONHASHSEED'] = "42"
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    X_train, Y_train,X_val,Y_val, X_test ,Y_test,nb_classes= generate_dataset(random_state)
    model=create_model(6,dropout,l2,lr,random_state,nb_classes)
    trained_model = train(X_train, Y_train,X_val,Y_val, X_test ,Y_test,nb_classes,epochs,batch_size,patience)
