# coding=utf-8
"""
Utility functions for building models.
"""
import pandas as pd
import numpy as np
import argparse
from collections import Counter
import math
import gc
import os
import sys
import psutil
import joblib
import json
from urllib.request import urlopen, quote
import requests
import dateutil


class Resample():
    """
    A class that preform data resample tasks, used
    as a parameter in cross_validation.

    Example:

    resampler = Resample(method='SMOTE', ratio=r)
    s, r, _, _ = cross_validation(X, y,
                                  LogisticRegression(C=10),
                                  resampler=resampler)

    Attributes:
    -----------
    method: str
        Resample method.
        Availabe method:
        'SMOTE': SMOTE algorithm for supersampling
        'under': undersampling, resample from majority
                 class

    raio: float
        The ratio of minority class to majority class.
    """

    def __init__(self, method='SMOTE', ratio=1, **kwargs):
        """
        Constructor.

        Parameters:
        -----------
        method: str
            Resample methods. Defaults to 'SMOTE'.

        ratio: float
            Number of samples in minority class over
            the number of samples in the majority
            class.

        **kwargs: keyward arguments
            Arguments for different methods:
            'SMOTE':
                random_state: int
                    Random state seed.
        """
        self.method = method
        self.ratio = ratio
        provided_methods = ['SMOTE', 'under']
        if method not in provided_methods:
            err_msg = 'Error: method not recognized.'
            err_msg += ' Only the following is provided:\n'
            print(err_msg)
            return
        if method == 'SMOTE':
            rs = SMOTE(ratio=ratio, **kwargs)
            self.rs = rs
        if method == 'under':
            return


    def fit_sample(self, X, y):
        
        """
        Resample the data.

        Parameters:
        -----------
        X: ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        
        y: ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns:
        --------
        X_res: ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.
        
        y_res: ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`
        """
        if self.method == 'SMOTE':
            X_res, y_res = self.rs.fit_sample(X, y)
        if self.method == 'under':
            cnt = Counter(y)
            size_expected = int(np.floor(cnt[1] / self.ratio))
            tmp = pd.DataFrame({'index': list(range(len(y))),
                                'label': y})
            good_idx = list(tmp.loc[tmp['label'] == 0, 'index'])
            bad_idx = list(tmp.loc[tmp['label'] == 1, 'index'])
            bad_X = X[bad_idx]
            bad_y = y[bad_idx]
            good_idx = sk_resample(good_idx, n_samples=size_expected)
            good_X = X[good_idx]
            good_y = y[good_idx]
            X_res = np.vstack((good_X, bad_X))
            y_res = np.append(good_y, bad_y)

        return X_res, y_res


def cross_validation(X, y, clf, n_fold=10, n_round=10, verbose=False, resampler=None):
    """
    Perform cross validation on the dataset with specified
    classifier.
    StratifiedKFold is used to devide data.
    In every round, the sample is suffled.

    Parameters:
    -----------
    X: dataframe or ndarray or list
        Data.

    y: array-like
        Label.

    n_fold: int
        Number of folds in each round of cross validation.
        Defaults to 10.

    n_round: int
        Number of rounds(times) to run the cv.
        Defaults to 10.

    verbose: bool
        Whether to plot addtional information.

    resampler: Resampler object
        A resampler object to perform resampling on
        training data in each split.

    Returns:
    --------
    stats: DataFrame
        Cross validation results.

    results: dict
        AUC and KS for training and testing.

    fig, axarr: matplotlib.Figure/Axes
        Plots handlers.
    """
    auc_train = []
    auc_test = []
    ks_train = []
    ks_test = []
    y = np.array(y)
    buffer = {'mean': {}, 'std': {}}
    for _ in range(n_round):
        idx = random.sample(list(np.arange(0, X.shape[0], 1)), X.shape[0])

        skf = StratifiedKFold(n_splits=n_fold, shuffle=True)
        d_x = X.iloc[idx, :]
        d_y = y[idx]

        for train_index, test_index in skf.split(d_x, d_y):
            X_train, X_test = d_x.iloc[train_index], d_x.iloc[test_index]
            y_train, y_test = d_y[train_index], d_y[test_index]

            if resampler is not None:
                X_train, y_train = resampler.fit_sample(X=X_train.as_matrix(), y=y_train)

            clf = clf.fit(X_train, y_train)

            p_test = clf.predict_proba(X_train)[:, 1]
            auc1 = roc_auc_score(y_true=y_train, y_score=p_test)
            ks1 = ks(y_true=create_y_true(y_train), y_proba=p_test, n_bin=50)

            p_test = clf.predict_proba(X_test)[:, 1]
            auc2 = roc_auc_score(y_true=y_test, y_score=p_test)
            ks2 = ks(y_true=create_y_true(y_test), y_proba=p_test, n_bin=50)

            auc_train.append(auc1)
            auc_test.append(auc2)
            ks_train.append(ks1)
            ks_test.append(ks2)

    buffer['mean']['auc-train'] = np.mean(auc_train)
    buffer['std']['auc-train'] = np.std(auc_train)
    buffer['mean']['auc-test'] = np.mean(auc_test)
    buffer['std']['auc-test'] = np.std(auc_test)
    buffer['mean']['ks-train'] = np.mean(ks_train)
    buffer['std']['ks-train'] = np.std(ks_train)
    buffer['mean']['ks-test'] = np.mean(ks_test)
    buffer['std']['ks-test'] = np.std(ks_test)
    stats = pd.DataFrame.from_dict(buffer, orient='index')
    stats = stats[['auc-train', 'auc-test', 'ks-train', 'ks-test']]
    results = {'auc_train': auc_train,
               'auc_test': auc_test,
               'ks_train': ks_train,
               'ks_test': ks_test}

    fig = None
    axarr = None
    
    if verbose:
        title_text = ['auc-train', 'auc-test', 'ks-train', 'ks-test']
        mean_floats = list(stats.iloc[0, :].values)
        std_floats = list(stats.iloc[1, :].values)
        print('           {} rounds {} folds cross-validation'.format(n_round, n_fold))
        print('-----------------------------------------------------------')
        print('        {:^12s}   {:^12s}   {:^12s}   {:^12s}'.format(*title_text))
        print('mean:  {:^12.4f}   {:^12.4f}   {:^12.4f}   {:^12.4f}'.format(*mean_floats))
        print('std:   {:^12.4f}   {:^12.4f}   {:^12.4f}   {:^12.4f}'.format(*std_floats))
        fig, axarr = plt.subplots(2, 2, figsize=(10,4))
        axarr[0][0] = pd.DataFrame({'train': auc_train, 'test': auc_test}).plot(ax=axarr[0][0])
        axarr[0][0].set_title('AUC-train')
        axarr[0][1] = sns.boxplot(auc_test, ax=axarr[0][1], orient='v', width=0.5)
        axarr[0][1].set_title('AUC-test')
        axarr[1][0] = pd.DataFrame({'train': ks_train, 'test': ks_test}).plot(ax=axarr[1][0])
        axarr[1][0].set_title('KS-train')
        axarr[1][1] = sns.boxplot(ks_test, ax=axarr[1][1], orient='v', width=0.5)
        axarr[1][1].set_title('KS-test')
        fig.tight_layout()

    return stats, results, fig, axarr


