# coding=utf-8
"""
Scripts for text analysis modeling.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from utils import ks
import utils
import joblib
from sklearn.model_selection import KFold
import random
from utils import create_y_true
from sklearn.svm import SVC
import time
from collections import Counter
from sklearn.metrics import roc_curve
import os
import sys
import psutil
import gc
import xgboost as xgb
import yaml
import argparse

def run_model(m, params_f='./params.txt', n_run=10,
              test_size=0.2,
              train_size1=0.8,
              train_size2=0.8,
              val_size=0.8):
    """
    Run models on the data.

    Parameters:
    -----------
    m: str
        Classifier to use.

    n_run: int
        Number of times to run the model
        on ramdonly sampled data.
        Defaults to 10.

    params_f: str
        File that stores parameters in yaml.
        Defaults to './params.txt'.

    test_size: float
        Test data size, the proportion in the
        entire data set.
        Defaults to 0.2.

    train_size1: float
        Train data size, the proportion in the
        entire data set, including actual training
        and validation set.
        Defaults to 0.8.

    train_size2: float
        Actual test data size, the proportion in
        train size1.
        Defaults to 0.8.

    val_size: float
        Validation data size, the proportion of data
        from the training data set (train_size1).
        Defaults to 0.2.
    """

    # models availabe
    print('Checking classifer ...')
    models = {'svm': SVC,
              'gbc': GradientBoostingClassifier,
              'lg': LogisticRegression,
              'adaboost': AdaBoostClassifier,
              'bnb': BernoulliNB, 
              'rf': RandomForestClassifier,
              'xgboost': xgb}

    if m not in models:
        print('Classifier not recoganized/supported.')
        print('Please check available classifiers.')
        print()
        return

    # load data
    print('Loading data ...')

    data = pd.read_csv('./data/model_data.csv', encoding='gb18030', index_col='贷款编号')

    # load key words
    key_word = pd.read_csv('./data/keyword.txt', encoding='gb18030', index_col=None, header=None, sep=' ')
    key_word = key_word.dropna(axis=1).iloc[0, :].values

    # split train and test data
    train_data, test_data = train_test_split(data, test_size=test_size, train_size=train_size1, random_state=7)

    print('Done.')
    print()



    if m != 'xgboost':

        # choose classifier
        clf = models[m]

        # load parameters if exists
        if os.path.exists(os.path.join(params_f)):
            with open(params_f, 'r') as f:
                params = yaml.load(f)
            if params is None:
                clf = clf()
            else:
                clf = clf(**params)
        else:
            clf = clf()

        # run model and save result
        aucs = []
        kss = []
        # models = []

        seed_used = []

        for i in range(n_run):
            curr_auc = []
            curr_ks = []

            # random seed
            seed = random.randint(1, 1000)
            while seed in seed_used:
                seed = random.randint(1, 1000)
            seed_used.append(seed)

            # split training and validation data
            dtrain, dval = train_test_split(data, test_size=val_size, train_size=train_size2, random_state=seed)

            # train model
            start_time = time.time()
            clf = clf.fit(X=dtrain[key_word], y=dtrain['label'])
            end_time = time.time()

            # training and validation, and testingr esults
            # training
            proba_train = clf.predict_proba(dtrain[key_word])[:, 1]
            y_true = create_y_true(dtrain['label'])

            auc = roc_auc_score(y_true=y_true, y_score=proba_train)
            ksv = ks(y_true=y_true, y_proba=proba_train)
            curr_auc.append(auc)
            curr_ks.append(ksv)
            
            # validation
            proba_val = clf.predict_proba(dval[key_word])[:, 1]
            y_true = create_y_true(dval['label'])

            auc = roc_auc_score(y_true=y_true, y_score=proba_val)
            ksv = ks(y_true=y_true, y_proba=proba_val)
            curr_auc.append(auc)
            curr_ks.append(ksv)

            # testing
            proba_test = clf.predict_proba(test_data[key_word])[:, 1]
            y_true = create_y_true(test_data['label'])

            auc = roc_auc_score(y_true=y_true, y_score=proba_test)
            ksv = ks(y_true=y_true, y_proba=proba_test)
            curr_auc.append(auc)
            curr_ks.append(ksv)

            aucs.append(curr_auc)
            kss.append(curr_ks)

            # models.append(clf)

            print('[{}]    training-auc: {:.7}    val-auc: {:.7}    test-auc: {:.7}    time: {:.3} min'.format(i,
                                                                                                               curr_auc[0],
                                                                                                               curr_auc[1],
                                                                                                               curr_auc[2],
                                                                                                               (end_time - start_time) / 60))

            gc.collect()

    # if xgboost is used
    else:
        aucs = []
        kss = []

        seed = random.randint(1, 1000)
        seed_used = [seed]

        dtrain, dval = train_test_split(train_data, test_size=val_size, train_size=train_size2, random_state=seed)
        # check for labels
        # if only one class is present in the data set, resample the data
        # implement this section later
        xgb_test_label = test_data['label']
        xgb_train = xgb.DMatrix(dtrain[key_word], label=dtrain.label)
        xgb_val = xgb.DMatrix(dval[key_word], label=dval.label)
        xgb_test = xgb.DMatrix(test_data[key_word])
        watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]

        # parameters
        params_found = True
        if os.path.exists(os.path.join(params_f)):
            with open(params_f, 'r') as f:
                params = yaml.load(f)
            if params is None:
                params_found = False
        else:
            params_found = False

        if not params_found:
            print('Parametes for xgboost is missing!')
            print('Program stops.')
            print()
            return

        num_rounds = params['num_rounds']
        early_stop = params['early_stop']
        parameter_list = list(params['params'].items())

        # start modeling
        start_time = time.time()
        model = xgb.train(parameter_list, xgb_train, num_rounds, watchlist, early_stopping_rounds=early_stop)
        end_time = time.time()
        print()
        print('Time spent: {:.3}'.format((end_time - start_time) / 60))
        print()

        curr_auc = []
        curr_ks = []
        p_test = model.predict(xgb.DMatrix(dtrain[key_word]))
        auc = roc_auc_score(y_true=create_y_true(dtrain.label), y_score=p_test)
        ksv = ks(y_true=create_y_true(dtrain.label), y_proba=p_test)

        curr_auc.append(auc)
        curr_ks.append(ksv)

        p_test = model.predict(xgb.DMatrix(dval[key_word]))
        auc = roc_auc_score(y_true=create_y_true(dval.label), y_score=p_test)
        ksv = ks(y_true=create_y_true(dval.label), y_proba=p_test)
        curr_auc.append(auc)
        curr_ks.append(ksv)

        y_pred = model.predict(xgb_test)
        auc = roc_auc_score(y_true=create_y_true(xgb_test_label), y_score=y_pred)
        ksv = ks(y_true=create_y_true(xgb_test_label), y_proba=y_pred)
        curr_auc.append(auc)
        curr_ks.append(ksv)

        aucs.append(curr_auc)
        kss.append(curr_ks)



    df_auc = pd.DataFrame(aucs, columns=['train', 'val', 'test'])
    df_ks = pd.DataFrame(kss, columns=['train', 'val', 'test'])

    # save results
    # create result dir if not present
    p = os.path.join('./data', 'results')
    if not os.path.exists(p):
        os.mkdir(p)

    # create model dir if not present
    p2 = os.path.join(p, m)
    if not os.path.exists(p2):
        os.mkdir(p2)
    else:
        c = 1
        p2 = os.path.join(p, '{}{}'.format(m, c))
        while os.path.exists(p2):
            c += 1
            p2 = os.path.join(p, '{}{}'.format(m, c))
        os.mkdir(p2)

    df_auc.to_csv(os.path.join(p2, 'auc.csv'))
    df_ks.to_csv(os.path.join(p2, 'ks.csv'))
    with open(os.path.join(p2, 'params.txt'), 'w') as f:
        yaml.dump(params, f)


def main():
    """
    Handle command line options.
    """
    parser = argparse.ArgumentParser()

    # commond
    parser.add_argument('-clf', '--classifier', required=True,
                        help='Classifer to use')

    parser.add_argument('-nr', '--num_run', default=10,
                        help='Number of times to run the model')

    parser.add_argument('-p', '--parameters', default='./params.txt',
                        help='File that stores parameters for the classifer')

    parser.add_argument('-ts1', '--train_size1', default=0.8, type=float,
                    help='train_size1')

    parser.add_argument('-ts2', '--train_size2', default=0.8, type=float,
                    help='train_size2')

    parser.add_argument('-ts', '--test_size', default=0.2, type=float,
                    help='test_size')

    parser.add_argument('-vs', '--val_size', default=0.2, type=float,
                    help='val_size')

    args = parser.parse_args()

    # run the model
    run_model(args.classifier, params_f=args.parameters, n_run=args.num_run,
              test_size=args.test_size, val_size=args.val_size,
              train_size1=args.train_size1, train_size2=args.train_size2)


if __name__ == '__main__':
    main()

