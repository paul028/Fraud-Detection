# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 23:19:41 2022

@author: paulv
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot

#sns.set_theme(style="darkgrid")
A4_DIMS = (11.7, 8.27)

def countplot(data, column, hue=None, log=False):
    top_n = data[column].value_counts().iloc[:20].index
    top = data[data[column].isin(top_n)]

    fig, ax = pyplot.subplots(figsize=A4_DIMS)
    ax.tick_params(labelrotation=90)
    if log: ax.set(yscale="log")
    sns.countplot(ax=ax, x=column, data=top, order=top[column].value_counts().index, hue=hue)
    
transactions = pd.read_csv("../data/transactions_train.csv")
transactions.head()

countplot(transactions, "type", log=True)
transactions["type"].value_counts()

countplot(transactions, "type", hue="isFraud", log=True)
transactions[transactions["isFraud"] == 1]["type"].value_counts()

countplot(transactions[transactions["isFraud"] == 0], "type", log=True)
transactions[transactions["isFraud"] == 0]["type"].value_counts()

countplot(transactions[transactions["isFraud"] == 1], "type")
transactions[transactions["isFraud"] == 1]["type"].value_counts()

countplot(transactions, "isFraud", log=True)
transactions["isFraud"].value_counts()

top_n = transactions.groupby("type").aggregate(func=np.median).sort_values(by="amount", ascending=False)
amount_type_df = transactions[["type", "amount"]]

fig, ax = pyplot.subplots(figsize=A4_DIMS)
ax.tick_params(labelrotation=90)
ax.set(yscale="log")
ax = sns.barplot(x="type", y="amount", data=amount_type_df, estimator=np.median, order=top_n.index)

top_n = transactions.groupby("type").aggregate(func=np.median).sort_values(by="amount", ascending=False)
amount_type_df = transactions[transactions["isFraud"] == 0][["type", "amount"]]

fig, ax = pyplot.subplots(figsize=A4_DIMS)
ax.tick_params(labelrotation=90)
ax.set(yscale="log")
ax = sns.barplot(x="type", y="amount", data=amount_type_df, estimator=np.median, order=top_n.index)

top_n = transactions.groupby("type").aggregate(func=np.median).sort_values(by="amount", ascending=False)
amount_type_df = transactions[transactions["isFraud"] == 1][["type", "amount"]]

fig, ax = pyplot.subplots(figsize=A4_DIMS)
ax.tick_params(labelrotation=90)
ax.set(yscale="log")
ax = sns.barplot(x="type", y="amount", data=amount_type_df, estimator=np.median, order=top_n.index)

top_n = transactions.groupby("type").aggregate(func=np.mean).sort_values(by="amount", ascending=False)
amount_type_df = transactions[transactions["isFraud"] == 1][["type", "amount"]]

fig, ax = pyplot.subplots(figsize=A4_DIMS)
ax.tick_params(labelrotation=90)
ax.set(yscale="log")
ax = sns.barplot(x="type", y="amount", data=amount_type_df, estimator=np.mean, order=top_n.index)

countplot(transactions, "step")
countplot(transactions[transactions["isFraud"] == 0], "step")
countplot(transactions[transactions["isFraud"] == 1], "step")

countplot(transactions[transactions["isFraud"] == 0], "nameOrig")
countplot(transactions[transactions["isFraud"] == 1], "nameOrig")
# Accounts with fraud transactions only transacted once

countplot(transactions[transactions["isFraud"] == 0], "nameDest")

countplot(transactions[transactions["isFraud"] == 1], "nameDest")