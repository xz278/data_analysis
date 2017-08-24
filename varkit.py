# coding=utf-8
"""
Variable analysis module.

Example:

test_df = pd.read_csv('./test_data/test_bining.csv',
                       encoding='gb18030',
                       index_col=None)

var_income = VarAnalysis(test_df,
                         var_col='INCOME',
                         label_col='label',
                         var_kind='n')

var_income.create_var_stats()

var_income.set_breakpoints([-0.01, 2000, 3000, 4000, math.inf])

var_income.pretty_df

var_income.plot_stats()


"""
import pandas as pd
import math
import numpy as np
import utils
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import dateutil
from dateutil import relativedelta as rdelta
from sklearn.tree import DecisionTreeClassifier
from collections import deque

class VarAnalysis():
    """
    A class to perform analysis on
    individual variables.
    """

    def __init__(self, data, var_col, var_kind, label_col='label',
                 pos_v=1, neg_v=0, bad_v=1):
        """
        Constructor.

        Parameters:
        -----------
        data: DataFrame
            Varable data and labels.

        var_col: str
            Variable column.

        var_kind: str
            Variable type.
            'numeric' or 'n' for numeric variable,
            'categorical' or 'c' for categorical variable.

        label_col: str
            Label column.
            Defaults to label.

        pos_v, neg_v: wildcard
            Value for positive and negative responses.
            Defaults are 1 and 0 respectively.

        bad_v: wildcard
            Value to use when calculating bad rate.
            Defaults to 1.
        """
        self.data = data[[var_col, label_col]]
        self.var_col = var_col
        self.var_kind = var_kind
        self.label_col = label_col
        self.pos_v = pos_v
        self.neg_v = neg_v
        self.bad_v = bad_v
        self.iv = None
        self.breakpoints = None

    def gen_breakpoints(self, verbose=False, **kwargs):
        """
        Generate breakpoints.
        """
        self.breakpoints = generate_breakpoints(self.data,
                                                var_col=self.var_col,
                                                label_col=self.label_col,
                                                verbose=verbose,
                                                **kwargs)

    def create_var_stats(self):
        """
        Group data and plot analysis stats.
        """
        self.data['bin'] = gen_bin(self.data,
                                   var_col=self.var_col,
                                   bins=self.breakpoints)
        # compute stats
        iv, stats_df, pretty_df = woe(self.data, verbose=True, pretty=True, s=False)
        self.iv = iv
        self.stats_df = stats_df
        self.pretty_df = pretty_df


    def set_breakpoints(self, bp):
        """
        Set breakpoints for variable binning.

        Parameters:
        -----------
        bp: list
            Breakpoints.
        """
        self.breakpoints = bp
        self.create_var_stats()


    def plot_stats(self, figsize=(20, 8)):
        """
        Plot grouped stats.
        """
        ax = woeplot(self.stats_df, var_name=self.var_col, figsize=figsize)
        return ax


def get_range(x):
    """
    Get the range of the data, i.e min and max.

    Example:

    x = [4, 1, 2, 5, 6, 2, 8]
    min_v, max_v = get_range(x)

    Parameters:
    -----------
    x: list/pandas.Series/array-like
        Data.

    Returns:
    --------
    l, r: float
        min and max of the data.
    """
    return min(x), max(x)


def to_breakpoints(intv, left=None, right=None, edge=None, data=None):
    """
    Convert a list of intervals to a list
    of breakpoints. NaN values are exclued.

    Example:

    intv = pd.cut([1,2,3,4,5,6,6,6,6,6,6], 3)
    bp = to_breakpoints(intv)
    bp = to_breakpoints(intv, edge='inf')

    Parameters:
    -----------
    intv: list of Interval objects.

    left, right: float
        Left most and right most edges of all intervals.
        Defaults to None.

    edge: str
        Type of left/right most egdes. When this parameter
        is set, left/right are discarded.
        'inf': -inf and inf for left and right most edges
        'range': min and max of 'data', edges are extended
                to 1.005 of range

    data: list
        Original data. Required when 'egde' is set to 'range'.

    Returns:
    --------
    bp: list
        List of breakpoints.
    """
    error_smg = '[!] Error: Parameter "data" is required'
    error_smg += ' when "edge" is set to "range".'
    error_smg += '\n    Check documentation.'
    bp = []
    c = 0
    intv = list(filter(lambda x: not pd.isnull(x), intv))
    uni_intv = sorted(pd.unique(intv))
    for i in uni_intv:
        if c == 0:
            bp.append(i.left)
            bp.append(i.right)
        else:
            bp.append(i.right)
        c += 1
    if edge is not None:
        if edge == 'inf':
            bp[0] = -math.inf
            bp[-1] = math.inf
        if edge == 'range':
            if data is None:
                print(error_smg)
                return []
            a, b = get_range(data)
            ext = (b - a) * 0.005
            bp[0] = a - ext
            bp[-1] = b + ext
    else:
        if left is not None:
            bp[0] = left
        if right is not None:
            bp[-1] = right
    return bp


def entropy(data):
    """
    Compute entropy of a set of data.

    Parameters:
    -----------
    data: list or numpy.array
        Data of categories.

    Returns:
    e: float
        Entropy.
    """
    # filter out nan value
    data = list(filter(lambda x: not pd.isnull(x), data))
    # get unique values, Counter, data size
    u = np.unique(data)
    cnt = Counter(data)
    s = len(data)
    # compute gini impurity
    e = 0
    for i in u:
        p_i = cnt[i] / s
        e -= p_i * math.log(p_i)
    return e


def generate_breakpoints(df, var_col, label_col, method='even_width',
            left=None, right=None, edge=None,
            verbose=False, **kwargs):
    """
    Generate bins.

    Parameters:
    -----------
    verbose: bool
        Whether to plot breakpoints.
        Defaults to False.
    """
    # even width bining
    if method in ['even_width', 'w']:
        intv = pd.cut(df[var_col], **kwargs)
        bp = to_breakpoints(intv, left, right, edge, df[var_col])

    # even depth binning
    if method in ['even_depth', 'd']:
        intv = pd.qcut(df[var_col], duplicates='drop', **kwargs)
        bp = to_breakpoints(intv, left, right, edge, df[var_col])

    # genearte bin automatically
    if method in ['woe', 'iv', 'gini', 'entropy']:
        tree = None
        tree = Tree(df, var_col=var_col, label_col=label_col)
        tree = tree.build(metric=method, **kwargs)
        bp = tree.get_breakpoints(left=left, right=right, edge=edge)
        tree = None

    if verbose:
        print(bp)

    return bp


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


def gen_bin(df, var_col, bins=10, method='even_width',
            left=None, right=None, edge=None, group=None,
            var_kind='n', verbose=False, **kwargs):
    """
    Generate bins for numerical and categorical variable.
    Group missing value in seperate bin.

    Example:

    test_df['bin'] = gen_bin(test_df, var_col='MAR_STATUS', var_kind='c',
                         group={'a': ['007005', '007003', '007002'],
                                'b': ['007001', '007004'],
                                'missing': ['missing']})

    test_df['bin'] = gen_bin(test_df, var_col='INCOME', bins=5, method='d', edge='inf')

    Parameters:
    -----------
    df: DataFrame
        Data.

    var_col: str
        Variable column.

    bins: int or list
        If int, it is number of bins.
        If list, it a list of breakpoint as
        input for pd.cut(), and other bining
        related parameters are discarded.
        Defaults to 10.

    method: str
        Type of bining methods.
        'even_width' or 'w': evenly spaced intervals
        'even_depth' or 'd': group data into euqal-size
                             subset
        'tree' or 't': tree based bining, use recursive
                       bining to find the best bin.
                       Create a maximun of 8 bins,
                       depth of 3.
        Defaults to 'w'

    left, right: float
        Left most and right most edges of all intervals.
        Defaults to None.

    edge: str
        Type of left/right most egdes. When this parameter
        is set, left/right are discarded.
        'inf': -inf and inf for left and right most edges
        'range': min and max of 'data', edges are extended
                to 1.005 of range

    var_kind: str
        Type varaible to analysis.
        'categorical' or 'c' for categorical variable.
        'numerical' or 'n' for numerical variable.
        Defaults to 'n'.

    group: dict
        Groups for categorical data.
        e.g. {'group_name': [group elements]}
        Defaults to None, use original categories as bins.

    verbose: bool
        Whether to print current binning.
        Defaults to False.

    **kwargs: keyword arguments
        Keyword arguments for generate bins automatically.

    Returns:
    --------
    bins_ret: list
        List of intervals corresponding to each row
        in the data.
        Missing or empty values are assign nan.
    """
    # for numerical variable
    if var_kind in ['numerical', 'n']:
        # if breakpoints are specified
        if type(bins) is list:
            bins_ret = pd.cut(df[var_col], bins=bins)
            return bins_ret

        # even width bining
        if method in ['even_width', 'w']:
            intv = pd.cut(df[var_col], bins)
            bp = to_breakpoints(intv, left, right, edge, df[var_col])
            bins_ret = pd.cut(df[var_col], bins=bp)

        # even depth binning
        if method in ['even_depth', 'd']:
            intv = pd.qcut(df[var_col], bins, duplicates='drop')
            bp = to_breakpoints(intv, left, right, edge, df[var_col])
            bins_ret = pd.cut(df[var_col], bins=bp)

        # use woe to genearte bin automatically
        if method == 'woe':
            print('???')

    # for categorical variable
    if var_kind in ['categorical', 'c']:
        if group is None:
            bins_ret = df[var_col]
        else:
            reverse_dict = {}
            for k in group:
                for v in group[k]:
                    reverse_dict[v] = k
            # check if new grouping is valid
            if set(reverse_dict.keys()) != set(df[var_col]):
                error_msg = '[!] Error: New grouping is not inclusive of all old categories.'
                error_msg += '\n    New grouping must include all old categories.'
                print(error_msg)
                bins_ret = []
            else:
                bins_ret = [reverse_dict[x] for x in df[var_col]]
    return bins_ret


def group_woe(df, tot_pos, tot_neg, label='label',
              pos_v=1, neg_v=0, bad_v=1, s=None,
              verbose=False):
    """
    Compute weight of evidence in
    current group of data.

    x = pd.DataFrame({'label': [1, 0,0,0,0,0]})
    w, iv = group_woe(x, tot_pos=50, tot_neg=50)
    w, iv, d = group_woe(x, tot_pos=50,
                         tot_neg=50, verbose=True)

    Parameters:
    -----------
    df: DataFrame
        Group data.

    label: str
        Label columns. Defaults to 'label'.

    tot_pos: int
        Total number of positive responses.

    tot_neg: int
        Total number of negative responses.

    pos_v, neg_v: wildcard
        Value for positive and negative responses.
        Defaults are 1 and 0 respectively.

    bad_v: wildcard
        Value to use when calculating bad rate.
        Defaults to 1.

    s: bool
        Smooth parameter used to avoid division by zero
        error when there is no negative responses in
        current group.
        Defaults to None, only use smoothing when
        there is only one kind of reponses in the data.
        When set to True, one instance of both
        positive and negative response will be added
        to data.

    verbose: bool
        Return verbose variable analysis.
        Defaults to False.

    Returns:
    --------
    w: float
        Weight of evidence.
        Return inf if there is no negative response,
        return -inf is there is no positive response.

    iv: float
        Information value.
        Return inf where there is only one type of
        responses in the data.

    d: dict
        Analysis of grouped data:
        'count': count of current group
        'tot_distr': percentage distribution total cases
        'positive': positive case count
        'pos_distr': positive distribution, pos / tot_pos
        'negative': negative case count
        'neg_distr': negative distribution, neg / tot_neg
        'bad_rate': bad rate, bad / group count
        'woe': weight of evident
        'iv': information value
    """
    if (s is None) and (df[label].unique().shape[0] == 1):
        s = 1
    elif s:
        s = 1
    else:
        s = 0
    num_pos = df.loc[df[label] == pos_v].shape[0] + s
    num_neg = df.loc[df[label] == neg_v].shape[0] + s
    if num_neg == 0:
        w = math.inf
        iv = math.inf
    elif num_pos == 0:
        w =  -math.inf
        iv = math.inf
    else:
        pos_distr = num_pos / tot_pos
        neg_distr = num_neg / tot_neg
        w = math.log(pos_distr / neg_distr)
        iv = (pos_distr - neg_distr) * w
    if verbose:
        tot = tot_pos + tot_neg
        d = {}
        d['count'] = df.shape[0]
        d['tot_distr'] = df.shape[0] / tot
        d['positive'] = num_pos - s
        d['pos_distr'] = (num_pos - s) / tot_pos
        d['negative'] = num_neg - s
        d['neg_distr'] = (num_neg - s) / tot_neg
        df_bad = df.loc[df[label] == bad_v]
        d['bad_rate'] = df_bad.shape[0] / df.shape[0]
        d['woe'] = w
        d['iv'] = iv
        return w, iv, d
    else:
        return w, iv


def woe(df, label='label', bin_col='bin',
        pos_v=1, neg_v=0, bad_v=1, s=None,
        verbose=False, var_kind='n', pretty=False):
    """
    Compute woe and related analysis of
    a variable.

    Example:
    
    iv, df, pretty_df = woe(test_df, verbose=True, pretty=True)

    Parameters:
    -----------
    df: DataFrame
        Variable data.

    label: str
        Label column. Defaults to 'label'.

    bin_col: str
        Bin column. Defaults to 'bin'.

    pos_v, neg_v, bad_v: wildcard
        Value for positive, negative response and
        bad rate values.

    s: bool
        Whether add a smoother in woe computation.

    verbose: bool
        Return verbose variable analysis.
        Defaults to False.

    var_kind: str
        Type varaible to analysis.
        'categorical' or 'c' for categorical variable.
        'numerical' or 'n' for numerical variable.
        Defaults to 'n'.

    pretty: bool
        Whether rename return dataframe columns for
        better readability.
        Defaults to False.

    Returns:
    --------
    iv: float
        Inforamtion Value.

    r_df: DataFrame
        Analsys for the variable if verbose is
        set to True.

    r_df_pretty: DataFrame
        A more readable table, if pretty is
        set to True.
    """
    cnt = Counter(df[label])
    tot_pos = cnt[pos_v]
    tot_neg = cnt[neg_v]
    
    # check variable type
    if var_kind not in ['c', 'categorical', 'numerical', 'n']:
        print('[!] Error: var_kind must be either "categorical" or "numerical"')
        return None, None, None

    buffer = {}
    # for numerical variable
    if var_kind in ['n', 'numerical']:
        if df[bin_col].isnull().any():
            tmp = df.loc[df[bin_col].isnull(), [label]]
            _, _, d = group_woe(df=tmp, bad_v=bad_v,
                                pos_v=pos_v, neg_v=neg_v,
                                verbose=True, s=s,
                                label=label,
                                tot_neg=tot_neg, tot_pos=tot_pos)
            d['_sort'] = -math.inf
            buffer['missing'] = d
        tmp = df.loc[~df[bin_col].isnull(), [label, bin_col]]
        for i, g in tmp.groupby(bin_col):
            _, _, d = group_woe(df=g, bad_v=bad_v,
                                pos_v=pos_v, neg_v=neg_v,
                                verbose=True, s=s,
                                label=label,
                                tot_neg=tot_neg, tot_pos=tot_pos)
            d['_sort'] = i.right
            buffer[str(i)] = d

    # for categorical variable
    if var_kind in ['c', 'categorical']:
        if 'missing' in df[bin_col].values:
            tmp = df.loc[df[bin_col] == 'missing', [label]]
            _, _, d = group_woe(df=tmp, bad_v=bad_v,
                                pos_v=pos_v, neg_v=neg_v,
                                verbose=True, s=s,
                                label=label,
                                tot_neg=tot_neg, tot_pos=tot_pos)
            d['_sort'] = -math.inf
            buffer['missing'] = d
        tmp = df.loc[df[bin_col] != 'missing', [label, bin_col]]
        for i, g in tmp.groupby(bin_col):
            _, _, d = group_woe(df=g, bad_v=bad_v,
                                pos_v=pos_v, neg_v=neg_v,
                                verbose=True, s=s,
                                label=label,
                                tot_neg=tot_neg, tot_pos=tot_pos)
            if d['woe'] == -math.inf:
                d['_sort'] = -9999999
            else:
                d['_sort'] = d['woe']
            buffer[i] = d

    # format data
    r_df = pd.DataFrame.from_dict(buffer, orient='index').sort_values('_sort').drop('_sort', axis=1)
    r_df = r_df[['count', 'tot_distr', 'negative', 'neg_distr',
                 'positive', 'pos_distr', 'bad_rate', 'woe', 'iv']]
    iv = r_df['iv'].sum()

    if pretty:
        r_df_pretty = r_df.copy()
        r_df_pretty.columns = ['Count', 'Total Distr', 'Goods',
                               'Distr Goods', 'Bads', 'Distr Bad',
                               'Bad Rate', 'WOE', 'IV']
        for c in ['Total Distr', 'Distr Goods', 'Distr Bad', 'Bad Rate']:
            r_df_pretty[c] = ['%.2f%%' % round(x, 2)
                              for x in r_df_pretty[c]]
        r_df_pretty['WOE'] = [round(x * 100, 3)
                              for x in r_df_pretty['WOE']]
        if 'missing' in r_df_pretty.index:
            tmp = r_df_pretty.index.values
            tmp[0] = 'Missing'
            r_df_pretty.index = tmp


    # output
    if verbose and pretty:
        return iv, r_df, r_df_pretty
    elif verbose:
        return iv, r_df
    else:
        return iv


def woeplot(df, figsize=(14, 5), var_name=None):
    """
    Plot WOE, data size, and bad rate by bins.

    Example:

    test_df = pd.read_csv('./test_data/test_bining.csv', encoding='gb18030', index_col=None)
    test_df['bin'] = gen_bin(test_df, var_col='INCOME', bins=5, method='d', edge='inf')
    iv, analysis, p = woe(test_df, verbose=True, pretty=True)
    fig, axarr = woeplot(analysis, var_name='Income')

    Parameters:
    -----------
    df: DataFrame
        Variable analysis data.

    var_name: str
        Variable name. Defaults to None.

    ax: matplotlib.axes.Axes
        Axes to draw the plot.

    figsize: tuple
        Figure size, defaults to (14, 5).

    Return:
    -------
    ax: matplotlib.axes.Axes
        Axes on which the plot is drawn.
    """
    df = df[['woe', 'bad_rate', 'count']]
    df['woe'] = df['woe'].apply(lambda x: 100 * x)
    df['bad_rate'] = df['bad_rate'].apply(lambda x: 100 * x)
    fig, axarr = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # WOE
    ax = df['woe'].plot(kind='bar', rot=0, color='darkblue', ax=axarr[0])
    n_bins = df.shape[0]
    ax.plot(np.arange(-1, n_bins + 1, 1), [0] * (n_bins + 2), '--k')
    ax.grid(axis='both')
    _ = ax.set_ylabel('WOE')

    # Bad rate
    ax = df['bad_rate'].plot(kind='bar', rot=0, color='darkred', ax=axarr[1])
    n_bins = df.shape[0]
    ax.grid(axis='both')
    if var_name is not None:
        _ = ax.set_xlabel(var_name)
    _ = ax.set_ylabel('Bad Rate')

    # Count
    ax = df['count'].plot(kind='bar', rot=0, color='darkgreen', ax=axarr[2])
    n_bins = df.shape[0]
    ax.grid(axis='both')
    if var_name is not None:
        _ = ax.set_xlabel(var_name)
    _ = ax.set_ylabel('Count')
    plt.subplots_adjust(left=0.1, bottom=0, right=0.9, top=0.94, hspace=0.2, wspace=0.3)
    
    if var_name is not None:
        _ = ax.set_xlabel(var_name)
    return fig, axarr


def split_feature(df, class_cnt, size, var_col, label_col, min_split_size,  metric='woe', **kwargs):
    """
    Use specified metric to determine whether or how
    to split the node.

    Parameters:
    -----------
    df: DataFrame
        Data.

    class_cnt: numpy.array
        Counts for class 0 and 1.

    size: int
        Size of current data.

    var_col, label_col: str
        Variable and label columns.

    min_split_size: int
        Minimum split size.

    metric: str
        Metric to split node.
        'gini': minimize gini impurity
        'entropy': maximize information gain
        'woe': maximize difference between weight of evidence
        'iv': maximize information value
        Defaults to 'woe'.

    kwargs:
        Keyword arguments for group_woe().

    Returns:
    --------
    bp: float or int
        Breakpoints. Return None when there
        is possible split.

    woe_left: float
        WOE for left child (smaller).

    woe_right: float
        WOE for right child (larger).

    min_split_size: int
        Minimun split size. Each subset must have at least
        this number of samples.
    """
    bp = None
    info_left = None
    info_right = None
    # return None if there is not enough data to split
    # in this node
    if (size >=4) and (class_cnt > 0).all():
    # search for the best breakpoint
        sorted_list = sorted(df[var_col].unique())

        # use WOE or IV to split feature
        if metric in ['woe', 'iv']:
            best_v = -1
            for i in sorted_list[:-1]:
                # check min_split_size parameter
                df_left = df.loc[df[var_col] <= i]
                df_right = df.loc[df[var_col] > i]
                if (df_left.shape[0] < min_split_size) or (df_right.shape[0] < min_split_size):
                    continue

                # compute woe for each subset
                woe_l, iv_l = group_woe(df=df_left, s=False, **kwargs)
                woe_r, iv_r = group_woe(df=df_right, s=False, **kwargs)

                # skip if there is only one class in any subset
                # information value of group of single class is inf
                if (iv_l == -math.inf) or (iv_r == -math.inf):
                    continue

                # compare and store best split
                if metric == 'woe':
                    v = abs(woe_l - woe_r)
                    if v > best_v:
                        best_v = v
                        bp = i
                        info_left = woe_l
                        info_right = woe_r

                elif metric == 'iv':
                    v = iv_l + iv_r
                    if v > best_v:
                        best_v = v
                        bp = i
                        info_left = iv_l
                        info_right = iv_r

        # gini impurity
        if metric == 'gini':
            best_v = math.inf
            for i in sorted_list[:-1]:
                # check min_split_size parameter
                df_left = df.loc[df[var_col] <= i]
                df_right = df.loc[df[var_col] > i]
                if (df_left.shape[0] < min_split_size) or (df_right.shape[0] < min_split_size):
                    continue
                
                gini_left = gini_impurity(df_left[label_col].values)
                gini_right = gini_impurity(df_right[label_col].values)

                # skip when gini = 0
                if (gini_left == 0) or (gini_right == 0):
                    continue

                # compare and store best split
                l_p = df_left.shape[0] / df.shape[0]
                r_p = df_right.shape[0] / df.shape[0]
                v = (gini_left * l_p + gini_right * r_p) / 2
                if v < best_v:
                    best_v = v
                    info_left = gini_left
                    info_right = gini_right
                    bp = i

        # entropy information gain
        if metric == 'entropy':
            best_v = -math.inf
            for i in sorted_list[:-1]:
                # check min_split_size parameter
                df_left = df.loc[df[var_col] <= i]
                df_right = df.loc[df[var_col] > i]
                if (df_left.shape[0] < min_split_size) or (df_right.shape[0] < min_split_size):
                    continue

                entropy_before = entropy(df[label_col].values)
                entropy_left = entropy(df_left[label_col].values)
                entropy_right = entropy(df_right[label_col].values)

                # skip when gini = 0
                if (entropy_left == 0) or (entropy_right == 0):
                    continue

                left_prop = df_left.shape[0] / df.shape[0]
                entropy_l = entropy_left * left_prop
                right_prop = df_right.shape[0] / df.shape[0]
                entropy_r = entropy_right * right_prop
                entropy_after = entropy_l + entropy_r
                v = entropy_before - entropy_after

                # compare and store best split
                if v > best_v:
                    best_v = v
                    info_left = entropy_left
                    info_right = entropy_right
                    bp = i

    return bp, info_left, info_right


def gini_impurity(data):
    """
    Compute gini impurity of a set of data.

    Parameters:
    -----------
    data: list or numpy.array
        Data of categories.

    Returns:
    g: float
        Gini impurity.
    """
    # filter out nan value
    data = list(filter(lambda x: not pd.isnull(x), data))
    # get unique values, Counter, data size
    u = np.unique(data)
    cnt = Counter(data)
    s = len(data)
    # compute gini impurity
    g = 0
    for i in u:
        p_i = cnt[i] / s
        g += p_i * (1 - p_i)
    return g


class TreeNode():
    """
    A node object for tree-based binning.
    Classes (target variable values) are 0 and 1.

    Attributes:
    -----------
    left: TreeNode
        left child node

    right: TreeNode
        right child node

    parent: TreeNode
        Parent node

    bp: float
        Breakpoint value, if None, then no split

    data: DataFrame
        Data in current node, include variable and
        label column.

    size: int
        Data size.

    class_cnt: numpy.array
        Counts for each class.

    value: float
        Value of metric for current node.

    var_col, label_col: str
        Variable and label columns in data.

    depth: int
        Depth of the node.
    """

    def __init__(self, data,
                 var_col, label_col,
                 value=None, bp=None,
                 depth=1, left=None,
                 right=None, parent=None,
                 is_leaf=False):
        """
        Constructor.

        Parameters:
        -----------
        data: DataFrame
            Data contained in current node.

        var_col, label_col: str
            Variable and label columns.

        depth: int
            Depth of the node. Defaults to 1 for
            root node.

        left: TreeNode
            left child node

        right: TreeNode
            right child node

        parent: TreeNode
            Parent node

        value: float
            Value of metric for current node.

        bp: float
            Breakpoint value, if None, then no split
        """
        self.data = data
        self.size = data.shape[0]
        cnt = Counter(data[label_col])
        self.class_cnt = np.array([cnt[0], cnt[1]])
        self.depth = depth
        self.left = left
        self.right = right
        self.parent = parent
        self.var_col = var_col
        self.label_col = label_col
        self.is_leaf = is_leaf
        self.bp = bp
        self.value = value


    def split(self,
              max_depth=math.inf,
              min_leaf_size=4,
              metric='woe',
              **kwargs):
        """
        Split the node if possible.

        Parameters:
        -----------
        max_depth: int
            Maximum depth of the tree, used for
            early stop. Defaults to inf.

        min_leaf_size: int
            Minimum leaf sample size.
        """
        # stop if reaches max_depth
        if self.depth > max_depth:
            self.set_leaf(True)
            return

        # stop if reaches min leaf size
        if self.size < min_leaf_size:
            self.set_leaf(True)
            return

        bp, info_left, info_right = split_feature(df=self.data,
                                                  var_col=self.var_col,
                                                  label_col=self.label_col,
                                                  class_cnt=self.class_cnt,
                                                  size=self.size,
                                                  metric=metric,
                                                  **kwargs)

        # if no further split, set the node to leaf node
        if bp is None:
            self.set_leaf(True)
        else:
            self.set_bp(bp)
            # create and split its descendants
            # left child
            self.left = TreeNode(self.data.loc[self.data[self.var_col] <= bp],
                                 var_col=self.var_col,
                                 label_col=self.label_col,
                                 depth=self.depth + 1,
                                 value=info_left,
                                 parent=self,
                                 is_leaf=False)
            self.left.split(max_depth=max_depth,
                            min_leaf_size=min_leaf_size,
                            metric=metric,
                            **kwargs)

            # right child
            self.right = TreeNode(self.data.loc[self.data[self.var_col] > bp],
                                 var_col=self.var_col,
                                 label_col=self.label_col,
                                 depth=self.depth + 1,
                                 value=info_right,
                                 parent=self,
                                 is_leaf=False)
            self.right.split(max_depth=max_depth,
                             min_leaf_size=min_leaf_size,
                             metric=metric,
                             **kwargs )


    def leaf(self):
        """
        Return True if this node is leaf,
        otherwise false.
        """
        return self.is_leaf


    def set_leaf(self, is_leaf):
        """
        Set the is_leaf attribute to specified value.

        Parameters:
        -----------
        is_leaf: bool
            Whether the node is a leaf node.
        """
        self.is_leaf = is_leaf


    def set_left(self, left):
        """
        Set left child.
        """
        self.left = left


    def set_right(self, left):
        """
        Set right child.
        """
        self.right = right


    def set_parent(self, parent):
        """
        Set parent child.
        """
        self.parent = parent


    def set_depth(self, depth):
        """
        Set depth of the node.
        """
        self.depth = depth


    def set_value(self, v):
        """
        Set metric value for current node.
        """
        self.value = v


    def set_bp(self, bp):
        """
        Set breakpoint.
        """
        self.bp = bp


    def get_bp(self):
        """
        Get breakpoint.
        """
        return self.bp


    def get_data(self):
        """
        Get data contained in current node.
        """
        return self.data


class Tree():
    """
    A tree object to perform tree-based binning.
    Initial data is stored in root.data.
    Classes should be 0 (non response) and 1 (response).

    Attributes:
    -----------
    metric: str
        Type of metrics to perform node spliting.
        'gini': gini impurity
        'entropy': information gain
        'woe': weight of evidence
        'variance': variance reduction.
        Currently only support woe.

    root: TreeNode
        Root node of the tree

    var_col, label_col: str
        Variable and label columns in data.

    size: int
        Sample size.
    """

    def __init__(self, data, var_col, label_col):
        """
        Constructor.
        
        Parameters:
        -----------
        var_col, label_col: str
            Variable and label columns in data.

        data: DataFrame
            Data.
        """
        self.var_col = var_col
        self.label_col = label_col
        data = data.sort_values(var_col)
        self.root = TreeNode(data, var_col, label_col)
        self.size = self.root.size

    def build(self,
              metric='woe',
              max_depth=math.inf,
              min_leaf_size=4,
              min_split_size=4):
        """
        Build the tree.

        Parameters:
        -----------
        metric: str
            Metric used to split nodes.

        max_depth: int
            Maximum depth of the tree, used for
            early stop. Defaults to inf.

        min_leaf_size: int or float
            If equal or greater than one, it is the minimum number
            of samples in one node.
            If smaller than one, it is the minimum proportion of
            sample size w.r.t total sample size.
            Minimum number of samples is 4 whatever the value is set.
            Defaults to 4.

        min_split_size: int or float
            Similar to min_leaf_size.
        """
        # validate parameters
        if max_depth < 1:
            print('[!] Error: Invalid max_depth:')
            print('    must be greater than 0.')
            return

        if (min_leaf_size <= 0) or (min_leaf_size >= self.size):
            print('[!] Error: Invalid min_leaf_size:')
            print('    must be in the range (0, n_sample).')
            return

        if (min_split_size <= 0) or (min_split_size >= self.size):
            print('[!] Error: Invalid min_split_size:')
            print('    must be in the range (0, n_sample).')
            return

        if min_leaf_size < 1:
            min_leaf_size = math.floor(self.size * min_leaf_size)
        min_leaf_size = max(4, min_leaf_size)

        if min_split_size < 1:
            min_split_size = math.floor(self.size * min_split_size)
        min_split_size = max(4, min_split_size)
        min_split_size = max(min_split_size, min_leaf_size)

        self.metric = metric

        cnt = Counter(self.root.data[self.label_col])
        tot_pos = cnt[1]
        tot_neg = cnt[0]
        self.root.split(max_depth=max_depth,
                        min_leaf_size=min_leaf_size,
                        min_split_size=min_split_size,
                        metric=metric,
                        tot_pos=tot_pos, tot_neg=tot_neg,
                        label=self.label_col)
        return self

    def prune():
        """
        Prune the tree.
        """
        return 0


    def tranverse():
        """
        Tranverse the tree.
        """
        return 0


    def __str__(self):
        """
        String form of this object.
        """
        s = ''
        return s

    def get_breakpoints(self, left=None, right=None, edge='range'):
        """
        Get breakpoints generated by the tree.

        Parameteres:
        ------------
        left, right: float
            Left most and right most edges of all intervals.
            Defaults to None.

        edge: str
            Type of left/right most egdes. When this parameter
            is set, left/right are discarded.
            'inf': -inf and inf for left and right most edges
            'range': min and max of 'data', edges are extended
                    to 1.005 of range

        Returns:
        --------
        bp: list
            List of breakpoints.
        """
        bp = self._get_breakpoints(self.root, [])

        if (edge is None) or (edge == 'range'):
            a, b = get_range(self.root.data[self.var_col])
            ext = (b - a) * 0.005
            l = a - ext
            r = b + ext

        if edge == 'inf':
            l = -math.inf
            r = math.inf

        if left is not None:
            l = left
        if right is not None:
            r = right
        bp = [l] + bp + [r]
        return bp


    def _breakpoints_by_depth(self):
        """
        Get breakpoints by depth of the tree.

        Returns:
        --------
        bp_by_depth: list
            
        """
        q = deque()
        q.append(self.root)
        bp_by_depth = []

        while len(q) > 0:
            tmp = []
            next_level = []
            for _ in range(len(q)):
                node = q.popleft()
                if not node.is_leaf:
                    tmp.append(node.bp)
                left = node.left
                if left is not None:
                    next_level.append(left)
                right = node.right
                if right is not None:
                    next_level.append(right)
            bp_by_depth.append(tmp)
            q.extend(next_level)

        return bp_by_depth
        


    @staticmethod
    def _get_breakpoints(node, bps):
        """
        A static method to extract breakpoints
        stored in the tree nodes by in-order tranversal.

        Parameters:
        -----------
        node: TreeNode

        bps: list
            Breakpoints.

        Returns:
        --------
        bps: list
            Breakpoints.
        """
        # return bps is node is null
        if node is None:
            return bps

        # tranverse left child
        bps = Tree._get_breakpoints(node.left, bps)
        # add breakpoint if node is not leaf node
        if not node.is_leaf:
            bps.append(node.bp)
        # tranverse right child
        bps = Tree._get_breakpoints(node.right, bps)

        return bps
