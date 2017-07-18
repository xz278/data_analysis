# coding=utf-8
"""
Utility functions for data analysis.
"""
import pandas as pd
import numpy as np
import argparse
import numpy as np
import pandas as pd
from collections import Counter
import numpy as np

def compute_woe(df, res_col, bin_col='bin', smooth=1):
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





def main():
    """
    Handle command line options.
    """
    return 0
