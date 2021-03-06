# coding=utf-8
"""
This module includes functions to perform techincal analysis,
including creating bins, arranging data, and ploting.

Example

dist_df: distance between work, res, live, sale addresses.
k = 'work-res'

>>> # plot the distributiion of data
>>> feature_distplot(df=dist_df, f_col=k)
>>>
>>> # automatically create bins
>>> bins = auto_bin(dist_df[k], kind='qcut', n=10, disp=True)
>>>
>>> # manually create bins
>>> bins = gen_bin(breakpoints=[-0.001, 1000, 6000, 12000, 28000,
>>>                             45000, 69000, 96000,
>>>                             173000, 3462699])
>>>
>>> # generate response data and group data
>>> r = gen_responsedata(dist_df[k],
>>>                      dist_df['label'],
>>>                      1,
>>>                      bins)
>>> g = gen_groupdata(r)
>>>
>>> # calculate and print iv value
>>> iv, _ = utils.compute_woe(df=r, bin_col='bin', res_col='y')
>>> print('IV: {:.6}'.format(iv))
>>> # a different way of compute iv
>>> iv = compute_iv_from_group(g)
>>>
>>> # plot response rate and data size in each bin
>>> fig, _, _ = plot_feature(g)
"""
import pandas as pd
import numpy as np
import yaml
from geopy.distance import vincenty
import utils
from collections import Counter
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats.stats as stats


def auto_bin(v, n, kind='qcut', disp=False):
    """
    Generate bins automatically (not optimal).
    #########################################
    #   Add optimal bining later.           #
    #########################################

    Parameters:
    -----------
    v: pandas.series
        Features.

    n: int
        Number of bins.

    kind: str
        Type of rules to create bin.
        Defaults to 'qcut', based on quantitles.
        'cut' create bins that are evenly spaced.

    disp: bool
        Whether to print the intervals.
        Defaults to False.

    Returns:
    --------
    bins: list
        List of Interval objects.
    """
    if kind == 'qcut':
        try:
            bins = np.unique(pd.qcut(v.dropna(), n))
        except:
            bins = np.unique(pd.qcut(v.dropna(), n, duplicates='drop'))
    else:
        bins = np.unique(pd.cut(v.dropna(), n))

    bins = sorted(bins)

    if disp:
        for i in bins:
            print(str(i))

    return bins



def feature_distplot(df, f_col, f=None, t=None):
    """
    Plot the distribution of the selected feature,
    and also display the max and min value.

    Parameters:
    -----------
    df: DataFrame
        Data.

    f_col: str
        Feature column.

    t: str
        Figure title.
        Defaults to None.

    f: str
        File to save the plot.
        Defaults to None.
        
    """
    ax = sns.distplot(df[f_col].dropna())
    if t is not None:
        ax.set_title(f_col)

    print('Range: {}, {}'.format(min(df[f_col]), max(df[f_col])))

    if f is not None:
        ax.figure.savefig(f)


def plot_feature(df, f=None, y_bar_scale=3):
    """
    Plot response rate of a feature
    in seperate bins.

    Parameters:
    -----------
    df: DataFrame
        Grouped data.

    f: str
        File name to store the plot.
        Defaults to None.

    Returns:
    --------
    fig: matplotlib.figure
        Figure handle of the plot.

    baraxis, curveaxis: matplotbib.axis
        Axis handle for bar plot and curve plot.
    """
    fig, curveaxis = plt.subplots(figsize=(10,4))
    baraxis = curveaxis.twinx()

    x_ticks = list(range(df.shape[0]))
    # plot overall postive rate
    opr = df['overall_pr'][0]
    curveaxis.plot([0] + x_ticks,
                   [opr for _ in range(len(x_ticks) + 1)],
                   '#6F6F6F')

    # curve plot
    curveaxis.plot(x_ticks, df['pr'], 'r')
    # tsax.xaxis.tick_top()

    # bar plot
    baraxis.bar(x_ticks, df['sum'], width=0.3, facecolor='#94CDF3')
    baraxis.bar(x_ticks, df['p'], width=0.3, facecolor='#F47E60')
    bar_y_limit = df['sum'].max() * y_bar_scale

    baraxis.set_ylim([0, bar_y_limit])
    max_y, min_y = max(df['sum']), min(df['sum'])
    y_ticks = np.arange(0, max_y * y_bar_scale, int(max_y * y_bar_scale / 7))
    # baraxis.set_yticks(y_ticks)
    # baraxis.set_yticks([0, bar_y_limit])
    # barax.xaxis.tick_top()

    # configure plot properties
    baraxis.set_ylabel('Sampel size')
    curveaxis.set_ylabel('Positive rate')
    _ = curveaxis.set_xticks(list(range(len(x_ticks))))
    _ = curveaxis.set_xticklabels(df.index, rotation=90)
    fig.tight_layout()
    if f is not None:
        fig.savefig(f)
    return fig, baraxis, curveaxis


def gen_groupdata(df):
    """
    Generate group data.

    Parameters:
    -----------
    df: DataFrame
        Response data.

    g_df: DataFrame
        Group data.
    """
    g_df = []

    # overall positive rate
    df_sum = df.shape[0]
    df_positive = df.loc[df['y']].shape[0]
    df_pr = df_positive / df_sum
    
    cols = ['bin', 'min', 'max', 'sum',
            'p', 'n', 'pr', 'nr', 'overall_pr']
    for b, group in df.groupby('bin'):
        entry = []
        l, r = endpoints(b)
        cnt = Counter(group['y'])
        entry.extend([b, l, r])
        p = cnt[True]
        n = cnt[False]
        s = p + n
        pr = p / s
        nr = n / s
        entry.extend([s, p, n, pr, nr, df_pr])
        g_df.append(entry)
        entry = None
    g_df = pd.DataFrame(g_df, columns=cols)
    g_df = g_df.set_index('bin')
    g_df = g_df.sort_values('min', ascending=True)
    return g_df


def gen_responsedata(x, y, true_v, bins):
    """
    Generate resoponse data.

    Parameters:
    -----------
    x: list
        Feature data.

    y: list:
        Response data.

    true_v: type of element in y
        True value of the response.

    bins: list
        List of intervals.

    Returns:
    --------
    df: DataFrame
        Grouped data.
    """
    df = pd.DataFrame({'x': x, 'y': [i == 1 for i in y], 'bin': assign(x, bins)})
    return df


def assign(d, bins):
    """
    Assign bins to data.
    Nan value is assigned with label 'missing'.

    Parameters:
    -----------
    d: array-like
        Data.

    bins: list of pandas.Interval
        Bins.
    """
    bin_list = []

    for v in d:
        if pd.isnull(v):
            bin_list.append('missing')
            continue

        assigned = False
        for b in bins:
            if v in b:
                bin_list.append(str(b))
                assigned = True
                break

        if not assigned:
            bin_list.append('missing')
    return bin_list



def gen_bin(breakpoints):
    """
    Generate a list of bins given a list
    of break points.
    Only works for continuous values.

    Parameters:
    -----------
    breakpoints: list
        List of break points, sorted in
        ascending order.

    Returns:
    --------
    bins: list
        Generated bins, a list of
        Pandas.Interval.
    """
    # return None if less than 3 breakpoints
    # is received
    if len(breakpoints) < 3:
        return None

    # generate breakpoints  
    s = 0
    bins = []
    for e in range(1, len(breakpoints)):
        if s >= e:
            print('Breakpoint Error: lefthand side({}) >= righthand side({})'.format(s, e))
            return None
        intv = pd.Interval(breakpoints[s], breakpoints[e])
        s = e
        bins.append(intv)
    return bins


def endpoints(intv_str):
    """
    Return the end points of a pandas.Interval str.
    """
    if intv_str == 'missing' or intv_str == 'nan':
        l = - (sys.maxsize - 100)
        r = l + 1
    else:
        l, r = [float(x) for x in intv_str[1:-1].split(',')]
    return l, r


def compute_iv_from_group(df, p_col='p', n_col='n', smooth=True):
    """
    #######################
    ###  Domain Error   ###
    #######################
    Compute IV value from grouped data.

    Parameters:
    -----------
    df: DataFrame
        Grouped data.

    p_col: str
        Positive response count column.
        Defaults to 'p'.

    n_col: str
        Negative response count column.
        Defaults to 'n'.

    smooth: bool
        Whether to smooth data to avoid
        extreme values.
        Defaults to True.

    Returns:
    --------
    iv: float
        IV value.
    """
    if smooth:
        smooth_v = 1
    else:
        smooth_v = 0
    n_bins = df.shape[0]
    all_pos = df[p_col].sum() + n_bins * smooth_v
    all_neg = df[n_col].sum() + n_bins * smooth_v
    iv = 0
    for _, row in df.iterrows():
        p = row[p_col] + smooth_v
        n = row[n_col] + smooth_v
        p_prob = p / all_pos
        n_prob = n / all_neg
        woe = math.log(p_prob / n_prob)
        iv += (p_prob - n_prob) * woe
    return iv


def mono_bin(X, Y, n=20):
    """
    An attempt to implement monotonic bining,
    from [https://statcompute.wordpress.com/
    2012/12/08/monotonic-binning-with-python/]
    ++++++++++++++++++++++++++++++++++++++++++
    +++         Not recommended!!!         +++
    ++++++++++++++++++++++++++++++++++++++++++

    Parameters:
    -----------
    X: pandas.Series
        Features.

    Y: pandas.Series
        Labels.

    Returns:
    --------
    list
        List of intervals

    d3: pandas.DataFrame
        Grouped data
    """
    X2 = X.fillna(np.median(X))
    r = 0
    while np.abs(r) < 1:
        d1 = pd.DataFrame({'X': X2, 'Y': Y, 'Bucket': pd.qcut(X2, n, duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index=True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
        d3 = pd.DataFrame(d2.min().X, columns = ['min_' + X.name])
        d3 = pd.DataFrame()
        d3['min_' + X.name] = d2.min().X
        d3['max_' + X.name] = d2.max().X
        d3[Y.name] = d2.sum().Y
        d3['total'] = d2.count().Y
        d3[Y.name + '_rate'] = d2.mean().Y
        d3.sort_values('min_' + X.name)
    return list(d3.index.values), d3


def group_categorical_data(df, k, na_col='missing'):
    """
    Reformate categorical feature and label
    into the required format for plot_feature.
    """
    df = df.replace([np.nan], na_col)
    overall_response_rate = df.loc[df['label'] == 1].shape[0] / df.shape[0]
    g_df = {}

    for i, group in df.groupby(k):
        e = {}
        cnt = Counter(group['label'])
        s = group.shape[0]
        p = cnt[1]
        n = cnt[0]
        pr = p / s
        nr = n / s
        e['sum'] = s
        e['p'] = p
        e['n'] = n
        e['pr'] = pr
        e['nr'] = nr
        e['overall_pr'] = overall_response_rate
        g_df[i] = e
        e = None
    g_df = pd.DataFrame.from_dict(g_df, orient='index')
    if na_col not in g_df.index:
        g_df = g_df.sort_values('pr', ascending=True)
    else:
        g_df['_sort'] = g_df['pr']
        g_df.loc[na_col, '_sort'] = -1
        g_df = g_df.sort_values('_sort', ascending=True)
        g_df = g_df.drop('_sort', axis=1)
    return g_df


def update_df(df):
    """
    Update Dataframe information.
    """
    df = df.rename(columns={'sum': '总人数',
                            'p': '坏客户', 'n': '好客户',
                            'pr': '坏账率', 'nr': '好账率',
                            'overall_pr': '总坏账率'})
    return df
