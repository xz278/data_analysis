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
    ax = sns.distplot(dist_df[k].dropna())
    if t is not None:
        ax.set_title(t)

    print('Range: {}, {}'.format(min(dist_df[k]), max(dist_df[k])))

    if f is not None:
        ax.figure.savefig(f)


def plot_feature(df, f=None):
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

    # curve plot
    x_ticks = list(range(df.shape[0]))
    curveaxis.plot(x_ticks, df['pr'], 'r')
    # tsax.xaxis.tick_top()

    # plot overall postive rate
    opr = df['overall_pr'][0]
    curveaxis.plot(x_ticks, [opr for _ in range(len(x_ticks))], 'k')

    # bar plot
    baraxis.bar(x_ticks, df['sum'], width=0.3, facecolor='b')
    bar_y_limit = df['sum'].max() * 3
    baraxis.set_yticks([0, bar_y_limit])
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
    if len(breakpoints) <= 3:
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