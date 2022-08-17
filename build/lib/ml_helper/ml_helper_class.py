#!/usr/bin/python3
"""

Author: Josef Koumar
e-mail: koumajos@fit.cvut.cz, koumar@cesnet.cz

Copyright (C) 2022 CESNET

LICENSE TERMS

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Neither the name of the Company nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

ALTERNATIVELY, provided that this notice is retained in full, this product may be distributed under the terms of the GNU General Public License (GPL) version 2 or later, in which case the provisions of the GPL apply INSTEAD OF those given above.

This software is provided as is'', and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the company or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
"""
# for file handling
import sys
import os

# for data handling
import pandas as pd
import numpy as np
import json
import collections

# for ploting
from matplotlib import pyplot as plt
import seaborn as sns

# for ml model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from sklearn.svm import SVC

# models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# error handler
from .error_handler import DatabaseFileError
from .error_handler import LabelError
from .error_handler import ColumnError


class MLHealper:
    def __init__(self, filename: str = ""):
        if os.path.isfile(filename) is not True:
            raise DatabaseFileError(f"File {filename} doesn't exists.")
        if filename.endswith(".csv") is not True:
            raise DatabaseFileError(f"File {filename} isn't CSV file.")

        self.df = pd.read_csv("database.csv")
        self.label = None
        self.label_code = None
        # test and train dataset
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # evaluation
        self.classifications_array = None
        # classifiers
        self.randomforest = None
        self.randomforest_pred = None

    def set_label(self, labelname: str = ""):
        if labelname == "":
            raise LabelError(f"The label name is not set.")
        if labelname not in self.df.columns:
            raise LabelError(f"The label name is not in dataframe.")

        self.label = labelname
        self.label_code = labelname + "_code"
        self.set_type([self.label], "category")
        self.df[self.label_code] = self.df[self.label].cat.codes

    def handle_null_values(self, column_names: list, value: str):
        for column_name in column_names:
            if column_name not in self.df.columns:
                raise ColumnError(f"The column name {column_name} is not in dataframe.")
            self.df.loc[self.df[column_name].isnull(), column_name] = value

    def set_type(self, column_names: list, type: str):
        for column_name in column_names:
            if column_name not in self.df.columns:
                raise ColumnError(f"The column name {column_name} is not in dataframe.")
            self.df[column_name] = self.df[column_name].astype(type)

    def labels_counts(self):
        return self.df[self.label].value_counts()

    def train_test_split(self, features: list, test_size: int = 0.30):
        X = self.df[features]
        y = self.df[self.label_code]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y
        )

    # evaluate results
    def get_label_names(self):
        real_class = []
        for i in self.y_test:
            if i not in real_class:
                real_class.append(i)

        classifications = {}
        df_class = self.df[[self.label, self.label_code]]
        for index, row in df_class.iterrows():
            if row[self.label_code] not in classifications:
                classifications[row[self.label_code]] = row[self.label]

        classifications_all = []
        self.classifications_array = []
        classifications = collections.OrderedDict(sorted(classifications.items()))
        for k, v in classifications.items():
            if k in real_class:
                self.classifications_array.append(str(v))
            classifications_all.append(v)
        print(self.classifications_array)

    def get_classification_report(self):
        if self.classifications_array is None:
            self.get_label_names()
        return classification_report(
            self.y_test, self.randomforest_pred, target_names=self.classifications_array
        )

    # ML models
    def RandomForestClassifier(self, n_estimators=100):
        self.randomforest = RandomForestClassifier(n_estimators=n_estimators)
        self.randomforest.fit(self.X_train, self.y_train)
        self.randomforest_pred = self.randomforest.predict(self.X_test)
        print(
            "Accuracy of RandomForestClassifier: ",
            metrics.accuracy_score(self.y_test, self.randomforest_pred),
        )
