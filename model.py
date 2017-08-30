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


class XgboostWrapper():
    """
    A wrapper class of Xgboost to resemble
    api provided by scikit-learn machien learning
    algorithm.

    Example:

    with open('./xgboost_params.txt', 'r') as f:
        params = yaml.load(f)
    clf = XgboostWrapper(list_params=params['params'], num_rounds=params['num_rounds'])
    clf = clf.fit(X=X_train, y=y_train)
    p_test = clf.predict_proba(X=X_test)[:, 1]
    auc_score = roc_auc_score(y_true=y_test, y_score=p_test)

    Attributes:
    ----------
    dtrain: xgboost.DMatrix
        Training data.

    feature_names_: list of str
        Features names.

    list_params: list
        List parameterse for xgboost.train()

    num_rounds: int
        Xgboost.train() number of iterations.

    kwargs: dict
        Parametes for xgboost.train()
    """

    def __init__(self, list_params, num_rounds, **kwargs):
        """
        Constructor.

        Parameters:
        -----------
        kwargs: keyword arguments
            keyword parameters for xgboost.train()
        """
        self.list_params = list_params
        self.num_rounds = num_rounds
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        Fits the data to the classifier.
        Similar to the fit funcion in scikit learn.

        Parameters:
        -----------
        X: DataFrame, numpy.ndarray or list
            Training data.

        y: array-like
            Labels.
        """
        self.dtrain = xgb.DMatrix(X, y)
        self.model = xgb.train(list(self.list_params.items()),
                               self.dtrain,
                               self.num_rounds,
                               **self.kwargs)
        self.feature_names_ = self.model.feature_names

        feature_importances_gain = self.model.get_score(importance_type='gain')
        gain = [feature_importances_gain[x] if x in feature_importances_gain else 0
                for x in self.feature_names_]
        self.feature_importances_ = np.array(gain)
        self.feature_gain_ = np.array(gain)

        feature_importances_cover = self.model.get_score(importance_type='cover')
        cover = [feature_importances_cover[x] if x in feature_importances_cover else 0
                for x in self.feature_names_]
        self.feature_cover_ = np.array(cover)

        feature_importances_weight = self.model.get_score(importance_type='weight')
        weight = [feature_importances_weight[x] if x in feature_importances_weight else 0
                for x in self.feature_names_]
        self.feature_weight_ = np.array(weight)
        return self

    def predict_proba(self, X):
        """
        Predict the probability of class 0 and 1.

        Parameters:
        -----------
        X: DataFrame, numpy.ndarray or list
            Testing data.
        """
        p_test = self.model.predict(xgb.DMatrix(X))
        predict_0 = (1 - p_test).reshape([-1, 1])
        predict_1 = p_test.reshape([-1, 1])
        pred = np.hstack([predict_0, predict_1])
        return pred


def confusion_matrix(label, pred, verbose=False):
    """
    Create a confusion matrix.
    Assume 1 for positive and 0 for negative.
    Addition information includes:
    'accuarcy': (tp + tn) / total
    'missclassification rate': (fp + fn) / total
    'true positive rate': tp / actual yes
        also known as 'sensitivity' or 'recall'
    'false positive rate': fp / actual no
    'specificity': tn / actual no
    'precision': tp / predicted yes
    'prevalence': actual yes / total

    label = [0,0,0,0,1,1,1,1,1,1,1,1]
    pred = [1,1,0,0,0,0,1,1,1,1,0,0]

    cm ,stats = confusion_matrix(label=label, pred=pred, verbose=True)

    Parameters:
    -----------
    label: list
        True labels.

    pred: list
        Predicted class, NOT PROBABILITIES.

    verbose: bool
        Whether print verbose results.
        Defautls to False.

    Returns:
    --------
    cm: pandas.DataFrame
        Confusion matrix.

    stats: dict
        Other statistics.
    """
    df = pd.DataFrame({'label': label, 'pred': pred})
    tp = df.loc[(df['label'] == 1) & (df['pred'] == 1)].shape[0]
    fp = df.loc[(df['label'] == 0) & (df['pred'] == 1)].shape[0]
    tn = df.loc[(df['label'] == 0) & (df['pred'] == 0)].shape[0]
    fn = df.loc[(df['label'] == 1) & (df['pred'] == 0)].shape[0]
    cm = pd.DataFrame([[tn, fp], [fn, tp]],
                      columns=['Predicted: NO', 'Predicted: YES'],
                      index=['Actual: NO', 'Actual: YES'])

    accuarcy = (tp + tn) / df.shape[0]
    misclassification = (fp + fn) / df.shape[0]
    tp_rate = tp / df.loc[df['label'] == 1].shape[0]
    fp_rate = fp / df.loc[df['label'] == 0].shape[0]
    specificity = tn / df.loc[df['label'] == 0].shape[0]
    precision = tp / df.loc[df['pred'] == 1].shape[0]
    prevalence = df.loc[df['label'] == 1].shape[0] / df.shape[0]
    stats = {'accuarcy': accuarcy,
             'misclassification rate': misclassification,
             'true positive rate': tp_rate,
             'sensitivity': tp_rate,
             'recall': tp_rate,
             'false positive rate': fp_rate,
             'specificity': specificity,
             'precision': precision,
             'prevalence': prevalence}

    if verbose:
        s = '               Confusion Matrix\n'
        s += '-------------------------------------------\n'
        s += str(cm) + '\n\n'
        s += '                  Statistics\n'
        s += '-------------------------------------------\n'
        cnt = Counter(df['label'])
        s_len = len(str(df.shape[0])) + 4
        s += '           {:>{}}{:>{}}{:>{}}\n'.format('Total', s_len,
                                                      'Yes', s_len,
                                                      'No', s_len)
        s += 'Acutal:    {:>{}}{:>{}}{:>{}}\n'.format(df.shape[0], s_len,
                                                      cnt[1], s_len,
                                                      cnt[0], s_len)
        cnt = Counter(df['pred'])
        s += 'Predicted: {:>{}}{:>{}}{:>{}}\n\n'.format(df.shape[0], s_len,
                                                        cnt[1], s_len,
                                                        cnt[0], s_len)
        s += 'TP: %d\n' % tp
        s += 'FP: %d\n' % fp
        s += 'TN: %d\n' % tn
        s += 'FN: %d\n' % fn
        s += 'Accuarcy:                   %.4f\n' % accuarcy
        s += 'Misclassification Rate:     %.4f\n' % misclassification
        s += 'Sensitivity/Recall/TP Rate: %.4f\n' % tp_rate
        s += 'Specificity:                %.4f\n' % specificity
        s += 'FP Rate:                    %.4f\n' % fp_rate
        s += 'Precision:                  %.4f\n' % precision
        s += 'Prevalence:                 %.4f\n' % prevalence
        s += '\n*****\n'
        s += 'Accuarcy: (TP + TN) / total\n'
        s += 'Sensitivity: FP / Actual yes\n'
        s += 'Specificity: TN / Actual no\n'
        s += 'Precision: TP / Predicted yes\n'
        s += 'Prevalence: Actual yes / total\n'
        print(s)
    return cm, stats


def rank_features(X, y, feature_pool, feature_type, clf=None, verbose=False):
    """
    Use backwards feature selection to select
    the best features.
    Remove the feature that brings the least effects
    to the overall performance of the model measured by auc.

    Examples:

    res = rank_features(pdl,
                        pdl.label,
                        feature_to_keep,
                        feature_type,
                        XgboostWrapper(list_params=params['params'],
                                       num_rounds=params['num_rounds']),
                        verbose=True)



    Parameters:
    ----------
    X: DataFrame
        Data.

    y: array-like
        Labels.

    feature_pool: list of str
        Features to choose from.

    feature_type: dict
        Type of features.
        {'numerical': [numerical var names],
         'categorical': [categorical var names]}

    clf: classifier object
        Classifier.
        Defautls to None, in which case
        logistic regression is used.
        fit() and predict_proba() will be called
        on the classifier object.

    verbose: bool
        Whether to plot the information.
        Defaults to False.

    Returns:
    --------
    res: DataFrame
        Results.
    """
    # benchmark auc score
    if clf is None:
        clf = LogisticRegression()
    pool = set(feature_pool)
    c_var = set(feature_type['categorical'])
    n_var = set(feature_type['numerical'])
    dtrain, _ = prepare_vars(X,
                             numerical_vars=list(n_var & pool),
                             categorical_vars=list(c_var & pool),
                             reset_index=True, scale=True)
    auc_train, auc_test , _, _ = cross_validation(dtrain, y, clf=clf, verbose=False)
    tot_auc_train = [np.mean(auc_train)]
    tot_auc_test = [np.mean(auc_test)]
    var_num = [len(feature_pool)]
    rank = []

    while len(pool) > 1:
        best_test = 0
        best_train = 0
        v = None
        for f in pool:
            curr_f = pool.copy()
            curr_f.remove(f)
            dtrain, _ = prepare_vars(X,
                                     numerical_vars=list(n_var & curr_f),
                                     categorical_vars=list(c_var & curr_f),
                                     reset_index=True, scale=True)
            auc_train, auc_test , _, _ = cross_validation(dtrain, y,
                                                          clf=clf,
                                                          verbose=False)
            auc_train = np.mean(auc_train)
            auc_test = np.mean(auc_test)
            if auc_test > best_test:
                best_test = auc_test
                best_train = auc_train
                v = f
        pool.remove(v)
        tot_auc_train.append(best_train)
        tot_auc_test.append(best_test)
        var_num.append(len(pool))
        rank.append(v)
    rank.append(list(pool)[0])
    res = pd.DataFrame({'auc-train': tot_auc_train,
                        'auc-test': tot_auc_test,
                        'num_var': var_num,
                        'rank': rank})
    if verbose:
        ax = res[['auc-train', 'auc-test', 'num_var']].plot(x='num_var', figsize=(14, 5), grid=True)
        ax.set_xlabel('Number of remaining features')
        ax.set_ylabel('AUC')
        _ = ax.set_xticks(np.arange(0, res.shape[0], 1))
        _ = ax.set_xticks(np.arange(res.shape[0], 0, -1))
    return res
