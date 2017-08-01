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
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import json

def run_model(params_f='./params.txt',
              config_f='./config.txt'):
    """
    Run models on the data.

    Parameters:
    -----------
    clf: str
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

    gs_param: str
        Grid search parameter to add to log.
        Defaults to None.
    """
    # load config file
    if os.path.exists(os.path.join(config_f)):
        with open(config_f, 'r') as f:
            configs = yaml.load(f)
    clf = configs['classifier']
    n_run = configs['num_run']
    testsize = configs['testsize']
    testseed = configs['testseed']
    trainsize = configs['trainsize']
    trainseed = configs['trainseed']
    valsize = configs['valsize']
    gs_param = configs['grid_search_params']
    gs_p_str = []
    for k in gs_param:
        gs_p_str.append(str(k) + ':' + str(gs_param[k]))
    gs_p_str = ';'.join(gs_p_str) 

    # models availabe
    print('Checking classifer ...')
    models = {'svm': SVC,
              'gbc': GradientBoostingClassifier,
              'lr': LogisticRegression,
              'adaboost': AdaBoostClassifier,
              'bnb': BernoulliNB, 
              'rf': RandomForestClassifier,
              'xgboost': xgb}

    if clf not in models:
        print('Classifier not recoganized/supported.')
        print('Please check available classifiers.')
        print()
        return

    # load data
    print('Loading data ...')
    data = pd.read_csv('./data/model_data.csv',
                       encoding='gb18030',
                       index_col='贷款编号')

    # load key words
    key_word = pd.read_csv('./data/keyword.txt',
                           encoding='gb18030',
                           index_col=None,
                           header=None,
                           sep=' ')
    key_word = key_word.dropna(axis=1).iloc[0, :].values

    # prepare training, validation, and testing data
    # check whether the data size specified is valid
    data['_select'] = list(range(data.shape[0]))
    if (trainsize + valsize + testsize) > data.shape[0]:
        print('[!] Train/val/testsize not valid.')
        return
    pn_cnt = Counter(data['label'])
    neg_rate = pn_cnt[0] / data.shape[0]
    neg_set = set(data.loc[data['label'] == 0]['_select'])
    pos_set = set(data.loc[data['label'] == 1]['_select'])

    # testing set
    neg_cnt = int(testsize * neg_rate)
    if neg_cnt == 0:
        neg_cnt += 1
    if neg_cnt > data.shape[0]:
        neg_cnt -= 1
    pos_cnt = testsize - neg_cnt
    test_neg_set = set(resample(list(neg_set),
                                n_samples=neg_cnt,
                                replace=False,
                                random_state=testseed))
    neg_set -= test_neg_set
    test_pos_set = set(resample(list(pos_set),
                                n_samples=pos_cnt,
                                replace=False,
                                random_state=testseed))
    pos_set -= test_pos_set
    test_set = test_neg_set | test_pos_set
    test_data = data.loc[data['_select'].isin(test_set)]

    clf_str = clf

    print('Training ...')
    if clf != 'xgboost':
        model_to_save = None
        best_auc = -1
        # choose classifier
        clf = models[clf]

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
        accs = []

        seed_used = []
        neg_set_bk = neg_set.copy()
        pos_set_bk = pos_set.copy()
        total_time = 0
        for i in range(n_run):
            neg_set = neg_set_bk.copy()
            pos_set = pos_set_bk.copy()
            curr_auc = []
            curr_ks = []
            curr_acc = []

            # random seed
            trainseed = random.randint(1, 1000)
            while trainseed in seed_used:
                trainseed = random.randint(1, 1000)
            seed_used.append(trainseed)
            seed_used.append(trainseed * 2)

            # split training and validation data
            # training set
            neg_cnt = int(trainsize * neg_rate)
            if neg_cnt == 0:
                neg_cnt += 1
            if neg_cnt > data.shape[0]:
                neg_cnt -= 1
            pos_cnt = trainsize - neg_cnt
            train_neg_set = set(resample(list(neg_set),
                                        n_samples=neg_cnt,
                                        replace=False,
                                        random_state=trainseed))
            neg_set -= train_neg_set
            train_pos_set = set(resample(list(pos_set),
                                        n_samples=pos_cnt,
                                        replace=False,
                                        random_state=trainseed))
            pos_set -= train_pos_set
            train_set = train_neg_set | train_pos_set
            dtrain = data.loc[data['_select'].isin(train_set)]

            # validation set
            neg_cnt = int(valsize * neg_rate)
            if neg_cnt == 0:
                neg_cnt += 1
            if neg_cnt > data.shape[0]:
                neg_cnt -= 1
            pos_cnt = valsize - neg_cnt
            val_neg_set = set(resample(list(neg_set),
                                        n_samples=neg_cnt,
                                        replace=False,
                                        random_state=trainseed))
            val_pos_set = set(resample(list(pos_set),
                                        n_samples=pos_cnt,
                                        replace=False,
                                        random_state=trainseed))
            val_set = val_neg_set | val_pos_set
            dval = data.loc[data['_select'].isin(val_set)]

            train_size_log = dtrain.shape[0]

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

            pred_train = clf.predict(dtrain[key_word])
            acc = accuracy_score(y_pred=pred_train, y_true=dtrain['label'])
            curr_acc.append(acc)

            # validation
            proba_val = clf.predict_proba(dval[key_word])[:, 1]
            y_true = create_y_true(dval['label'])

            auc = roc_auc_score(y_true=y_true, y_score=proba_val)
            ksv = ks(y_true=y_true, y_proba=proba_val)
            curr_auc.append(auc)
            curr_ks.append(ksv)

            pred_val = clf.predict(dval[key_word])
            acc = accuracy_score(y_pred=pred_val, y_true=dval['label'])
            curr_acc.append(acc)

            # testing
            proba_test = clf.predict_proba(test_data[key_word])[:, 1]
            y_true = create_y_true(test_data['label'])

            auc = roc_auc_score(y_true=y_true, y_score=proba_test)
            ksv = ks(y_true=y_true, y_proba=proba_test)
            curr_auc.append(auc)
            curr_ks.append(ksv)

            pred_test = clf.predict(test_data[key_word])
            acc = accuracy_score(y_pred=pred_test, y_true=test_data['label'])
            curr_acc.append(acc)


            aucs.append(curr_auc)
            kss.append(curr_ks)
            accs.append(curr_acc)

            # save the model with the highest testing auc
            if model_to_save is None:
                model_to_save = clf
                best_auc = aucs[-1][-1]
            elif aucs[-1][-1] > best_auc:
                model_to_save = clf
                best_auc = aucs[-1][-1]

            time_spent = end_time - start_time
            total_time += time_spent
            minutes, seconds = divmod(time_spent, 60)
            print_info = '[{}]    training-auc: {:.7}    '
            print_info += 'val-auc: {:.7}    test-auc: {:.7}    '
            print_info += 'time: {:3}min {:.4}sec'
            print(print_info.format(i,
                                    curr_auc[0],
                                    curr_auc[1],
                                    curr_auc[2],
                                    minutes,
                                    seconds))

            # collect potential unreferenced variable to save memory
            gc.collect()

        df_auc = pd.DataFrame(aucs, columns=['train', 'val', 'test'])
        df_ks = pd.DataFrame(kss, columns=['train', 'val', 'test'])
        df_acc = pd.DataFrame(accs, columns=['train', 'val', 'test'])

        # save results
        # create result dir if not present
        p = os.path.join('./data', 'results')
        if not os.path.exists(p):
            os.mkdir(p)

        # create model dir if not present
        p2 = os.path.join(p, clf_str)
        if not os.path.exists(p2):
            os.mkdir(p2)
        else:
            c = 1
            p2 = os.path.join(p, '{}{}'.format(clf_str, c))
            while os.path.exists(p2):
                c += 1
                p2 = os.path.join(p, '{}{}'.format(clf_str, c))
            os.mkdir(p2)

        # save results
        df_auc.to_csv(os.path.join(p2, 'auc.csv'))
        df_ks.to_csv(os.path.join(p2, 'ks.csv'))
        df_acc.to_csv(os.path.join(p2, 'acc.csv'))
        with open(os.path.join(p2, 'params.txt'), 'w') as f:
            yaml.dump(params, f)

        with open(os.path.join(p2, 'config.txt'), 'w') as f:
            yaml.dump(configs, f)

        # save average performance
        tmp1 = []
        tmp2 = []
        tmp3 = []
        cols = ['train', 'val', 'test']
        for c in cols:
            tmp1.append(df_auc[c].mean())
            tmp2.append(df_ks[c].mean())
            tmp3.append(df_acc[c].mean())
        df_p = pd.DataFrame([tmp1, tmp2, tmp3], columns=cols)
        df_p['metrics'] = ['auc', 'ks', 'acc']
        df_p = df_p.set_index('metrics')
        df_p.to_csv(os.path.join(p2, 'performance.csv'))

        # save model
        joblib.dump(model_to_save, os.path.join(p2, 'model.dat'))

        # save log
        auc_list = df_p.loc['auc'].values
        ks_list = df_p.loc['ks'].values
        acc_list = df_p.loc['acc'].values

        log_msg = ','.join(['{}'] * 16)
        log_msg += '\n'
        total_time /= n_run
        m, s = divmod(total_time, 60)
        log_info_msg = log_msg.format(trainsize,
                                      clf_str,
                                      auc_list[0],
                                      auc_list[1],
                                      auc_list[2],
                                      ks_list[0],
                                      ks_list[1],
                                      ks_list[2],
                                      acc_list[0],
                                      acc_list[1],
                                      acc_list[2],
                                      '{}:{}'.format(int(m), int(s)),
                                      p2,
                                      '{}'.format(os.path.join(p2,
                                                  'params.txt')),
                                      '{}'.format(os.path.join(p2,
                                                  'config.txt')),
                                      gs_p_str)
        with open('./data/results/log.txt', 'a') as f:
            f.write(log_info_msg)

    # if xgboost is used
    else:
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

        aucs = []
        kss = []
        accs = []

        if trainseed == -1:
            trainseed = random.randint(1, 1000)

        # split training and validation data
        # training set
        trainsize += valsize
        neg_cnt = int(trainsize * neg_rate)
        if neg_cnt == 0:
            neg_cnt += 1
        if neg_cnt > data.shape[0]:
            neg_cnt -= 1
        pos_cnt = trainsize - neg_cnt
        train_neg_set = set(resample(list(neg_set),
                                    n_samples=neg_cnt,
                                    replace=False,
                                    random_state=trainseed))
        neg_set -= train_neg_set
        train_pos_set = set(resample(list(pos_set),
                                    n_samples=pos_cnt,
                                    replace=False,
                                    random_state=trainseed))
        pos_set -= train_pos_set
        train_set = train_neg_set | train_pos_set
        dtrain = data.loc[data['_select'].isin(train_set)]

        # cross validation
        xgb_test_label = test_data['label']
        xgb_train = xgb.DMatrix(dtrain[key_word], label=dtrain.label)
        xgb_val = xgb.DMatrix(test_data[key_word], label=test_data.label)
        xgb_test = xgb.DMatrix(test_data[key_word])
        res = xgb.cv(params['params'],
                     xgb_train,
                     num_boost_round=params['num_rounds'],
                     nfold=n_run,
                     seed=trainseed,
                     stratified=True,
                     early_stopping_rounds=params['early_stop'],
                     verbose_eval=1,
                     show_stdv=True)

        # train model
        # num_rounds = params['num_rounds']
        # parameter_list = list(params['params'].items())
        # watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
        # # start modeling
        # start_time = time.time()
        # model = xgb.train(parameter_list,
        #                   xgb_train,
        #                   num_rounds,
        #                   watchlist,
        #                   early_stopping_rounds=params['early_stop'])
        # end_time = time.time()
        # print()
        # minutes, seconds = divmod(end_time - start_time, 60)
        # print('Time spent: {:3}min {:.4}sec'.format(minutes, seconds))
        # print()

        curr_auc = []
        curr_ks = []
        curr_acc = []

        # p_test = model.predict(xgb.DMatrix(dtrain[key_word]),
        #                        ntree_limit=model.best_iteration)
        # ksv = ks(y_true=create_y_true(dtrain.label), y_proba=p_test)

        curr_auc.append(res['train-auc-mean'][res.shape[0] - 1])
        # curr_ks.append(ksv)
        curr_ks.append(np.nan)
        curr_acc.append(np.nan)

        # validation results
        curr_auc.append(res['test-auc-mean'][res.shape[0] - 1])
        curr_ks.append(np.nan)
        curr_acc.append(np.nan)

        # test results
        # y_pred = model.predict(xgb_test,
        #                        ntree_limit=model.best_iteration)
        # auc = roc_auc_score(y_true=create_y_true(xgb_test_label), y_score=y_pred)
        # ksv = ks(y_true=create_y_true(xgb_test_label), y_proba=y_pred)
        # curr_auc.append(auc)
        # curr_ks.append(ksv)
        curr_auc.append(np.nan)
        curr_ks.append(np.nan)
        curr_acc.append(np.nan)

        aucs.append(curr_auc)
        kss.append(curr_ks)
        accs.append(curr_acc)

        # model_to_save = model
        gc.collect()

        df_auc = pd.DataFrame(aucs, columns=['train', 'val', 'test'])
        df_ks = pd.DataFrame(kss, columns=['train', 'val', 'test'])
        df_acc = pd.DataFrame(accs, columns=['train', 'val', 'test'])

        # save results
        # create result dir if not present
        p = os.path.join('./data', 'results')
        if not os.path.exists(p):
            os.mkdir(p)

        # create model dir if not present
        p2 = os.path.join(p, clf_str)
        if not os.path.exists(p2):
            os.mkdir(p2)
        else:
            c = 1
            p2 = os.path.join(p, '{}{}'.format(clf_str, c))
            while os.path.exists(p2):
                c += 1
                p2 = os.path.join(p, '{}{}'.format(clf_str, c))
            os.mkdir(p2)

        # save results
        df_auc.to_csv(os.path.join(p2, 'auc.csv'))
        df_ks.to_csv(os.path.join(p2, 'ks.csv'))
        with open(os.path.join(p2, 'params.txt'), 'w') as f:
            yaml.dump(params, f)

        # save average performance
        tmp1 = []
        tmp2 = []
        tmp3 = []
        cols = ['train', 'val', 'test']
        for c in cols:
            tmp1.append(df_auc[c].mean())
            tmp2.append(df_ks[c].mean())
            tmp3.append(df_acc[c].mean())
        df_p = pd.DataFrame([tmp1, tmp2, tmp3], columns=cols)
        df_p['metrics'] = ['auc', 'ks', 'acc']
        df_p = df_p.set_index('metrics')
        df_p.to_csv(os.path.join(p2, 'performance.csv'))

        # save model
        # joblib.dump(model_to_save, os.path.join(p2, 'model.dat'))

        # save cv results
        res.to_csv(os.path.join(p2, 'res.csv'))

        # save log
        auc_list = df_p.loc['auc'].values
        ks_list = df_p.loc['ks'].values
        acc_list = df_p.loc['acc'].values

        log_msg = ','.join(['{}'] * 16)
        log_msg += '\n'
        log_info_msg = log_msg.format(trainsize - valsize,
                                      clf_str,
                                      auc_list[0],
                                      auc_list[1],
                                      auc_list[2],
                                      ks_list[0],
                                      ks_list[1],
                                      ks_list[2],
                                      acc_list[0],
                                      acc_list[1],
                                      acc_list[2],
                                      # '{}:{}'.format(int(minutes),
                                      #                int(seconds)),
                                      'none',
                                      p2,
                                      '{}'.format(os.path.join(p2,
                                                  'params.txt')),
                                      '{}'.format(os.path.join(p2,
                                                  'config.txt')),
                                      gs_p_str)
        with open('./data/results/log.txt', 'a') as f:
            f.write(log_info_msg)


def main():
    """
    Handle command line options.
    """
    parser = argparse.ArgumentParser()

    # commond
    parser.add_argument('-p', '--parameters', default='./params/params.txt',
                        help='File that stores parameters for the classifer')

    parser.add_argument('-c', '--config', default='./params/config.txt',
                        help='Configurations for the scripts')

    args = parser.parse_args()

    # run the model
    run_model(config_f=args.config,
              params_f=args.parameters)


if __name__ == '__main__':
    main()
