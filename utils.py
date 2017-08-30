# coding=utf-8
"""
Utility functions for data analysis.
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
# import joblib
import json
from urllib.request import urlopen, quote
import requests
import dateutil

def compute_woe(df, res_col, bin_col='bin', smooth=1, auto_gc=True):
    """
    Compute WOE(weight of eveidence).

    Parameters:
    -----------
    df: DataFrame
        Data.

    res_col: str
        Response column, all values should be boolean
        variables.

    bin_col: str
        Bin columns, which is used to put data
        into different groups.

    smooth: float
        Smooth value to avoid inf WOE value.
        Default value is 1.

    auto_gc: boolean
        Whether collect garbage memory automatically.
        Defaults to True.
        Might be useful to Ubuntu.

    Returns:
    --------
    total_iv: float
        Total inforamtion value.

    woe_iv_dict: dict
        IV and WOE value for each bin, can be
        used to create pandas.DataFrame.
    """
    df = df.loc[~pd.isnull(df[res_col])]  #remove nan values from data
    total_pos = len(df.loc[df[res_col]])   # total number of positives
    total_neg = len(df) - total_pos        # total number of negatives
    
    
    # convert to bin column to str
    df[bin_col] = [str(x) for x in df[bin_col]]

    # compute WOE, IV for each bin
    woe_iv_dict = {}
    total_iv = 0
    for b, group in df.groupby(bin_col):
        curr_pos = len(group.loc[group[res_col]])  # current possitives
        curr_neg = len(group) - curr_pos        # current negatives

        # proportion of positives/negatives to total
        # positives/negatives
        # bin to total
        prop_pos = (curr_pos + smooth) / (total_pos + 2 * smooth)
        prop_neg = (curr_neg + smooth) / (total_neg + 2 * smooth)

        # woe
        curr_woe = math.log(prop_pos / prop_neg)
        # iv
        curr_iv = (prop_pos - prop_neg) * curr_woe
 
        woe_iv_dict[b] = ({'woe': curr_woe,
                     'iv': curr_iv})

        # add to total IV
        total_iv += curr_iv

    if auto_gc:
        gc.collect()

    return total_iv, woe_iv_dict


def ks(y_true, y_proba, n_bin=20):
    """
    Compute KS value, which is an indicator of
    the power of the classifier to discriminate
    differernt classes.

    Parameteres:
    ------------
    y_true: list of boolean variables
        Actual labels.

    y_proba: list of float
        List of probabilites of being positive.

    n_bin: int
        Number of bins to use.
        Default number is ten.
    """
    df = pd.DataFrame()
    df['y_true'] = y_true
    df['y_proba'] = y_proba
    df = df.sort_values('y_proba', ascending=True)
    bins = divide_list(list(df.y_proba), n_bin)
    ks = -1
    for l in bins:
        th = l[-1]
        tpr, fpr = tpr_fpr(list(df.y_true),
                           list(df.y_proba), th)
        diff = tpr - fpr
        if diff > ks:
            ks = diff
    return abs(ks)


def divide_list(data, n):
    """
    Divide data into n subsets.

    Parameters:
    -----------
    data: list
        Data

    n: int
        Number of subsets.

    Returns:
    --------
    """
    data = list(data)
    l = len(data)
    s = l / n
    ret = []
    p = 0
    f = np.floor(s)
    if f == 0:
        ret = [[x] for x in data]
        return ret
    if s - f < 0.5:
        s = int(f)
        for i in range(n):
            subset = data[p: p + s]
            p += s
            if i == (n - 1):
                subset.extend(data[p:])
            ret.append(subset)
            if len(subset) == 0:
                break
    else:
        s = int(f + 1)
        for i in range(n):
            if i == (n - 1):
                subset = (data[p:])
            else:
                subset = data[p: p + s]
                p += s
            if len(subset) == 0:
                break
            ret.append(subset)
    return ret


def tpr_fpr(y_true, y_proba, th):
    """
    Compute true positive rate and false positive rate.

    Parameters:
    -----------
    proba_col: list of floast
        Probability of being true, used to
        compute hypothetical class.

    y_true: list of boolean
        True class column.

    th: float
        Threshold probability.
    """
    y_h = [True if x >= th else False
           for x in y_proba]
    cnt = Counter(y_true)
    P = cnt[True]
    N = cnt[False]
    # true positive rate
    tp = list((x[0] and x[1] for x in zip(y_h, y_true)))
    tpr = Counter(tp)[True] / P
    # false positive rate
    fp = list(x[0] and not x[1] for x in zip(y_h, y_true))
    fpr = Counter(fp)[True] / N
    return tpr, fpr


def create_y_true(y, true_v=1):
    """
    Convert a list of 0's and 1's to True and False,
    based one given true value.

    Parameters:
    -----------
    y: list of int
        A list of labels.

    true_v: int
        The value representing True.
        Defualt is 1.

    Returns:
    --------
    y_true: list of bool
        True/False list.
    """
    y_true = [x == true_v for x in y]
    return y_true


def get_date_from_loan_no(loan_no_list):
    """
    Get date from loan number.
    No time zone information used.
    Time information would be handle should it
    is required in the future.

    Loan number should follow the format "110002016072704101"
    exactly, where date information is stored from
    position 5 to 12.

    Only one exception is met yet, which is "2016072704101".
    In this case, date inforamtion is stored from position 0
    to 8.

    Parameters:
    -----------
    loan_no_list: list of str
        Loan number list.

    Returns:
    --------
    date_list: list of datetime
        List of data, corresponding to each loan number.
    """
    # return empty list if loan no list is empty
    if len(loan_no_list) == 0:
        return []

    # check if loan no is in str format
    # if not, convert it to str
    if type(loan_no_list[0]) != str:
        loan_no_list = [str(x) for x in loan_no_list]

    # convert str to datetime object
    date_list = []
    p = -1
    for n in loan_no_list:
        p += 1
        l = len(n)
        if (l != 18) and (l != 13):
            print('Unrecognized format found at position {}:'.format(p))
            print('      {}'.format(n))
            return date_list
        if (l < 18):
            d = '{}-{}-{}'.format(n[:4], n[4: 6], n[6: 8])
        else:
            d = '{}-{}-{}'.format(n[5: 9], n[9: 11], n[11: 13])
        date_list.append(d)
    date_list = list(pd.to_datetime(date_list))
    return date_list


def save_paras(paras, f):
    """
    Convenient function to save model parameters
    to file in json.

    Parameters:
    -----------
    paras: dict
        Parameters to save.

    f: str
        File name.
    """
    with open(f, 'w') as outputfile:
        json.dump(paras, outputfile)
    print('Parameters saved to: "{}"'.format(f))


def load_paras(f):
    """
    Convenient function to load model parameters
    from a json file.

    Parameters:
    -----------
    f: str
        The file that stores the parameters.

    Returns:
    --------
    paras: dict
        Parameters.
    """
    with open(f, 'r') as inputfile:
        paras = json.load(inputfile)
    return paras



def keyword_appearance(df, fields, words):
    """
    Compute keyword appearance in the text fields.

    Parameters:
    -----------
    df: DataFrame
        Text data.

    fields: list of str
        Fields to look at.

    words: list of str
        Key words to look at.

    Returns:
    --------
    ret: DataFrame
        Statistics.
    """
    ret = {}

    # for each key word
    p = -1
    for w in words:
        p += 1
        entry = {}
        entry['sort'] = p
        # bad loan rate for its presence / absence
        df_presence = df.loc[df[w] == 1]
        if len(df_presence) == 0:
            entry['出现|坏账率'] = np.nan
        else:
            df_bad = df_presence.loc[df_presence['label'] == 1]
            r = len(df_bad) / len(df_presence)
            entry['出现|坏账率'] = r

        df_absence = df.loc[df[w] == 0]
        if len(df_absence) == 0:
            entry['出现|坏账率'] = np.nan
        else:
            df_bad = df_absence.loc[df_absence['label'] == 1]
            r = len(df_bad) / len(df_absence)
            entry['未出现|坏账率'] = r

        # appearance of the word in different fields
        sum_cnt = 0
        for f in fields:
            try:
                df_f = df_presence.loc[[False if pd.isnull(x)
                                        else w in x
                                        for x in df_presence[f]]]
            except:
                print(df_presence[f])
                print(w, f)
                return None
            field_cnt = len(df_f)
            entry[f] = field_cnt
            sum_cnt += field_cnt

        ret[w] = entry

    ret = pd.DataFrame.from_dict(ret, orient='index')
    ret = ret.sort_values('sort')
    ret = ret.drop('sort', axis=1)

    return ret


def sizeof(v, disp=False):
    """
    Get the size of a variable in MB.

    Parameters:
    -----------
    v: a variable or a pointer
        A variable

    disp: bool
        Whether print current memory to screen.
        Defaults to False.

    Returns:
    --------
    s: float
        The size of the variable in MB.
    """
    s = sys.getsizeof(v) / 1024 / 1024
    if disp:
        print('current memory: {:.3f} MB'. format(s))
    return s


def memory_now(disp=False):
    """
    Get the memroy cosumed by current python
    process.

    Parameters:
    -----------
    disp: bool
        Whether print current memory to screen.
        Defaults to False.

    Returns:
    --------
    m: float
        The size of memory taken up by current
        process in MB.
    """
    info = psutil.virtual_memory()
    m = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    if disp:
        print('current memory: {:.3f} MB'. format(m))
    return m


def get_latlon(text_address, ak=None):
    """
    Get latitude and longitude of the address through
    Baidu Map API geocoder.

    Parameters:
    -----------
    text_address: str
        Address in text format.

    ak: str
        Baidu Map API key.

    Returns:
    --------
    ret: dict
        Latitude, longitude, and other information
        returned by the api.
        Returned information includes,
        'status': 0 for success, otherwise failure,
        in the case of succes:
        'lat': latitdue,
        'lon': longitude,
        'precise': whether location is precise,
                   0 for precise, 1 otherwise,
        'confidence': level of confidence,
        'level': type of address
        In the case of bad request:
        'message': error message returned by server,
        'status': type of error.
    """
    # Use default key if ak is not provided
    if ak is None:
        ak = 'F4P8cT0VGfgvdDIVW9EvbnpscYvwbwTz'

    # type of errors
    # error_type = {1: '内部服务器错误',
    #               2: '请求参数非法',
    #               3: '权限校验失败',
    #               4: '配额校验失败',
    #               5: 'ak不存在或者非法',
    #               101: '服务禁用',
    #               102: '不通过白名单或者安全码不对',
    #               'other': {'2': '无权限',
    #                         '3': '配额错误'}}

    # service url
    url = 'http://api.map.baidu.com/geocoder/v2/'
    # output format
    output = 'json'
    # use quote to decode address to avoid unrecognized characters
    add = quote(text_address)
    uri = url + '?' + 'address=' + add  + '&output=' + output + '&ak=' + ak

    # maximum number of requests for url request timeout
    max_try = 10
    for i in range(max_try):
        try:
            req = urlopen(uri)
            break
        except:
            if i < max_try - 1:
                continue
            else:
                return {'succeed': False}

    res = req.read().decode()
    temp = json.loads(res)
    status = temp['status']
    # return lat, lon if request is successful,
    # otherwise return nan and print error message
    ret = {}
    if status == 0:
        res = temp['result']
        ret['lat'] = res['location']['lat']
        ret['lon'] = res['location']['lng']
        # ret['confidence'] = res['confidence']
        # ret['level'] = res['level']
        # ret['precise'] = res['precise']
        ret['succeed'] = True
    else:
        # ret = temp
        ret['succeed'] = False
    return ret


def batch_latlon(ls_addr, stop_length=61, stop_thr=10000, ak=None):
    """
    Retrieve latitude and longitude of a list of address.
    Will stop for 61 seconds (unless otherwise specified)
    after every 10000 request.

    Parameters:
    -----------
    ls_addr: list of str
        List of text address.

    stop_length: int
        Seconds to wait before starting new request.
        Defaults to 61 seconds.

    stop_thr: int
        Number of requests in a time interval.

    ak: str
        Baidu api key.
        Defaults to None.

    Returns:
    --------
    ls_latlon: list of tuples
        List of gps coordinates.
    """
    c = 0
    ls_latlon = []
    for a in ls_addr:
        c += 1
        if c == stop_thr:
            c = 0
            time.sleep(stop_length)
        if ak is None:
            r = get_latlon(text_address=a)
        else:
            r = get_latlon(text_address=a, ak=ak)
        if r['succeed']:
            ls_latlon.append((r['lat'], r['lon']))
        else:
            ls_latlon.append((np.nan, np.nan))
    return ls_latlon


def check_length(l):
    """
    Check the length of all elements in
    a list.

    Parameters:
    -----------
    l: list
        List of elements to check

    Returns:
    --------
    r: collections.Counter
        Counter of the length.
    """
    r = Counter([len(str(x)) for x in l])
    return r


def cons_count(l, v=1):
    """
    Compute the number of most recent and highest
    consecutive occurance of a given value.

    Parameters:
    -----------
    l: list
        List/Array-like data.

    v: type of element in l
        Value to consider.

    Returns:
    c: int
        Current consecutive.

    m: int
        Maximum consecutive.
    """
    # current
    p = len(l)
    c = 0
    while (p - 1 >= 0) and (l[p - 1] == v):
        c += 1
        p -= 1

    # maximum
    m = -1
    curr_m = 0
    p = len(l)
    while p - 1 >= 0:
        if l[p - 1] == v:
            curr_m += 1
        else:
            if curr_m > m:
                m = curr_m
            curr_m = 0
        if (p - 1 == 0) and (curr_m > m):
            m = curr_m
        p -= 1
    return c, m


def plot_roc(y_true, y_score, f=None):
    """
    Draw the roc curve.

    y_true: list of bool
        True label.

    y_score: list of floats
        Prediction scores for each label.

    f: str
        File to store the results.
        Defaults to None, do not save.
    """
    fpr, tpr, _ = roc_curve(y_test, p_test)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    if f is not None:
        plt.savefig(f)


def divide_list(l, n):
    """
    Divide list l into n subsets in ascending order.
    Nan value is excluded from all subsets.

    Parameters:
    -----------
    l: list
        List of data.

    n: int
        Number of subsets. Must be smaller than or
        equal to the length of the list.

    Returns:
    --------
    buckest: list (n_buckets, buckets_size)
        List of elements for each group.
    """
    if n > len(l):
        print('[!] n must be smaller than or equal to the number of elements in the list.')
        return
    i, r = divmod(len(l), n)

    bucket_size = [i for _ in range(n)]
    p = 0
    for j in range(r):
        bucket_size[j] += 1
        p += 1
    buckets = []
    p = 0
    for s in bucket_size:
        buckets.append(l[p: p+s])
        p += s
    return buckets


def rmonth(m):
    """
    Return a relative delta of month m.
    """
    return dateutil.relativedelta.relativedelta(months=m)


def count_nan(l):
    """
    Return the number of nan values
    in the list.
    """
    s = 0
    for i in l:
        if pd.isnull(i):
            s += 1
    return s


def outlier_threshold(x):
    """
    Compute threshold for detecting outliers
    in series of numeric values.
    Outliers are points outside of
    [(Q1 - 1.5 * IQR), (Q3 + 1.5 * IQR)],
    where Q1 and Q3 are 25% and 75% quantiles
    and IQR is (Q3 - Q1).

    Parameteres:
    ------------
    x: Pandas.Series
        Data.

    lower, upper: float
        Lower and upper threholds for out liers.
    """
    # filter out nan values
    x = list(filter(lambda z: not pd.isnull(z), x))
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def main():
    """
    Handle command line options.
    """
    return 0
