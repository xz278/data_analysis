# coding=utf-8
"""
This module contains a class to help generate
data descriptoin.
"""
import pandas as pd
import numpy as np
import joblib
import dateutil
from collections import Counter
import gc
import matplotlib.pyplot as plt
import seaborn as sns
# pd.options.mode.chained_assignment = None 


class DataDescriber():
    """
    A class to descibe data.
    A date column containing Timestamp objects
    must be included and specified when a class
    is created.

    ***************************************************
    NOTE:
    1. A dataframe containing feature information must
       be specified before featuren analysis.

    2. Label column must be specified to plot boxplot
       for numerical values and bad rate for categorical
       values.
    """

    def __init__(self, data, date_col, feature_info=None, label_col=None):
        """
        Constructor.

        Parameters:
        -----------
        data: DataFrame
            Data.

        date_col: str
            Date column.

        feature_info: DataFrame
            Feature information.
        """
        self.data = data.copy()
        self.date_col = date_col
        self.daily_cnt = None
        self.color = tuple(np.array((82, 127, 194)) / 255)
        self.monthdist = None
        self.feature_info = feature_info
        self.label_col = label_col


    def dailydistplot(self, color=None, xtickinterval=10,
                      figsize=(20, 5), rot=0, logy=False,
                      **kwargs):
        """
        Compute and plot daily distribution.

        Parameters:
        -----------
        ax: matplotlib.axis
            The axis to plot the figure.

        color: 
            Color of the plot.

        figsize: tuple
            Figure size arguments for pd.DataFrame.plot().

        rot: float
            Rotation of x-axis ticklabels.
            Defaults to 0, i.e horizontal.

        logy: bool
            Whether to use log y axis.
            Defaults to False.

        kwargs: keyword arguments
            Arguments for pandas.dataframe.plot().

        Returns:
        --------
        ax: matplot.axis
            Axis of the plot.
        """
        if self.daily_cnt is None:
            min_date, max_date = min(data[self.date_col]), max(data[self.date_col])
            self.daterange = (min_date, max_date)
            d = min_date
            daily_cnt = []
            date_list = []
            while d <= max_date:
                daily_data = data.loc[data[date_col] == d]
                s = daily_data.shape[0]
                date_list.append(d)
                daily_cnt.append(s)
                d += pd.to_timedelta('1d')
            daily_cnt = pd.DataFrame({'Date': pd.date_range(min_date, max_date),
                                      'Daily Count': daily_cnt})
            daily_cnt['Date'] = [x.date() for x in daily_cnt['Date']]
            daily_cnt = daily_cnt.set_index('Date')
            daily_cnt = daily_cnt.sort_index()
            self.daily_cnt = daily_cnt

        if color is None:
            color = self.color

        ax = DataDescriber._distplot(data=self.daily_cnt,
                                     target_col='Daily Count',
                                     title='Daily Distribution',
                                     xlabel='Date',
                                     xtickinterval=xtickinterval,
                                     logy=logy,
                                     figsize=figsize,
                                     rot=rot,
                                     color=color,
                                     **kwargs)
        return ax


    def cumulativedistplot(self, color=None,
                           width=1, figsize=(20, 5),
                           xtickinterval=10, rot=0,
                           logy=False, **kwargs):
        """
        Cumulative daily distribution.

        Parameters:
        -----------
        ax: matplotlib.axis
            The axis to plot the figure.

        color: 
            Color of the plot.

        width: float
            Width of the plot.
            Defautls to 1.

        figsize: tuple
            Figure size arguments for pd.DataFrame.plot().

        xtickinterval: int
            X tick interval.

        rot: float
            Rotation of x-axis ticklabels.
            Defaults to 0, i.e horizontal.

        logy: bool
            Whether to use log y axis.
            Defaults to False.

        kwargs: keyword arguments
            Arguments for pandas.dataframe.plot().

        Returns:
        --------
        ax: matplot.axis
            Axis of the plot.
        """
        if self.daily_cnt is None:
            min_date, max_date = min(data[self.date_col]), max(data[self.date_col])
            self.daterange = (min_date, max_date)
            d = min_date
            daily_cnt = []
            date_list = []
            while d <= max_date:
                daily_data = data.loc[data[date_col] == d]
                s = daily_data.shape[0]
                date_list.append(d)
                daily_cnt.append(s)
                d += pd.to_timedelta('1d')
            daily_cnt = pd.DataFrame({'Date': pd.date_range(min_date, max_date),
                                      'Daily Count': daily_cnt})
            daily_cnt['Date'] = [x.date() for x in daily_cnt['Date']]
            daily_cnt = daily_cnt.set_index('Date')
            daily_cnt = daily_cnt.sort_index()
            self.daily_cnt = daily_cnt

        if 'Cumulative Count' not in self.daily_cnt.columns.values:
            daily_cml = []
            tmp = 0
            for i in self.daily_cnt['Daily Count']:
                tmp += i
                daily_cml.append(tmp)
            self.daily_cnt['Cumulative Count'] = daily_cml

        if color is None:
            color = self.color

        ax = DataDescriber._distplot(data=self.daily_cnt,
                                     target_col='Cumulative Count',
                                     title='Cumulative Daily Distribution',
                                     xlabel='Date',
                                     xtickinterval=xtickinterval,
                                     logy=logy,
                                     figsize=figsize,
                                     rot=rot,
                                     color=color,
                                     width=width,
                                     **kwargs)
        return ax


    def daily_to_csv(self, f_path):
        """
        Save daily and cumulative daily distribution to
        a csv file.

        Parameters:
        -----------
        f_path: str
            File name.

        Returns:
        --------
        """
        if self.daily_cnt is None:
            min_date, max_date = min(data[self.date_col]), max(data[self.date_col])
            self.daterange = (min_date, max_date)
            d = min_date
            daily_cnt = []
            date_list = []
            while d <= max_date:
                daily_data = data.loc[data[date_col] == d]
                s = daily_data.shape[0]
                date_list.append(d)
                daily_cnt.append(s)
                d += pd.to_timedelta('1d')
            daily_cnt = pd.DataFrame({'Date': pd.date_range(min_date, max_date),
                                      'Daily Count': daily_cnt})
            daily_cnt['Date'] = [x.date() for x in daily_cnt['Date']]
            daily_cnt = daily_cnt.set_index('Date')
            daily_cnt = daily_cnt.sort_index()
            self.daily_cnt = daily_cnt

        if 'Cumulative Count' not in self.daily_cnt.columns.values:
            daily_cml = []
            tmp = 0
            for i in self.daily_cnt['Daily Count']:
                tmp += i
                daily_cml.append(tmp)
            self.daily_cnt['Cumulative Count'] = daily_cml

        self.daily_cnt.to_csv(f_path, encoding='gb18030')


    def monthdistplot(self, color=None, figsize=(20, 5),
                      xtickinterval=None, rot=0,
                      logy=False, **kwargs):
        """
        Monthly distribution plot.

        Parameters:
        -----------
        Returns:
        --------
        """
        if self.monthdist is None:
            data['_month'] = [x.year*100 + x.month for x in data[date_col]]
            min_month, max_month = min(data['_month']), max(data['_month'])
            self.monthrange = min_month, max_month
            months = []
            m = min_month
            while m <= max_month:
                months.append(m)
                m_y, m_m = divmod(m, 100)
                m_m += 1
                if m_m > 12:
                    m_m = 1
                    m_y += 1
                m = m_y * 100 + m_m

            monthdist = []
            for m in months:
                df = data.loc[data['_month'] == m]
                monthdist.append(df.shape[0])
            monthdist = pd.DataFrame({'Month': months, 'Monthly Count': monthdist})
            monthdist = monthdist.set_index('Month')
            self.monthdist = monthdist

        ax = DataDescriber._distplot(data=self.monthdist,
                                     target_col='Monthly Count',
                                     title='Monthly Distribution',
                                     xlabel='Month',
                                     xtickinterval=xtickinterval,
                                     logy=logy,
                                     figsize=figsize,
                                     rot=rot,
                                     color=color,
                                     **kwargs)
        return ax


    def cumulativemonthdistplot(self, color=None, figsize=(20, 5),
                                xtickinterval=None, rot=0,
                                logy=False, **kwargs):
        """
        Monthly distribution plot.

        Parameters:
        -----------

        Returns:
        --------
        """
        if self.monthdist is None:
            data['_month'] = [x.year*100 + x.month for x in data[date_col]]
            min_month, max_month = min(data['_month']), max(data['_month'])
            self.monthrange = min_month, max_month
            months = []
            m = min_month
            while m <= max_month:
                months.append(m)
                m_y, m_m = divmod(m, 100)
                m_m += 1
                if m_m > 12:
                    m_m = 1
                    m_y += 1
                m = m_y * 100 + m_m

            monthdist = []
            for m in months:
                df = data.loc[data['_month'] == m]
                monthdist.append(df.shape[0])
            monthdist = pd.DataFrame({'Month': months, 'Monthly Count': monthdist})
            monthdist = monthdist.set_index('Month')
            self.monthdist = monthdist

        if 'Cumulative Monthly' not in self.monthdist.columns.values:
            s = 0
            cml = []
            for x in self.monthdist['Monthly Count']:
                s += x
                cml.append(s)
            self.monthdist['Cumulative Monthly'] = cml

        ax = DataDescriber._distplot(data=self.monthdist,
                                     target_col='Cumulative Monthly',
                                     title='Cumulative Monthly Distribution',
                                     xlabel='Month',
                                     xtickinterval=xtickinterval,
                                     logy=logy,
                                     figsize=figsize,
                                     rot=rot,
                                     color=color,
                                     **kwargs)
        return ax


    def monthly_to_csv(self, f_path):
        """
        Save monthly distribution to a csv file.
    
        Parameters:
        f_path: str
            File path to save the data.
        """
        if self.monthdist is None:
            data['_month'] = [x.year*100 + x.month for x in data[date_col]]
            min_month, max_month = min(data['_month']), max(data['_month'])
            self.monthrange = min_month, max_month
            months = []
            m = min_month
            while m <= max_month:
                months.append(m)
                m_y, m_m = divmod(m, 100)
                m_m += 1
                if m_m > 12:
                    m_m = 1
                    m_y += 1
                m = m_y * 100 + m_m

            monthdist = []
            for m in months:
                df = data.loc[data['_month'] == m]
                monthdist.append(df.shape[0])
            monthdist = pd.DataFrame({'Month': months, 'Monthly Count': monthdist})
            monthdist = monthdist.set_index('Month')
            self.monthdist = monthdist

        if 'Cumulative Monthly' not in self.monthdist.columns.values:
            s = 0
            cml = []
            for x in self.monthdist['Monthly Count']:
                s += x
                cml.append(s)
            self.monthdist['Cumulative Monthly'] = cml

        self.monthdist.to_csv(f_path, encoding='gb18030')


    def distplot(self, target_col,
                 title=None,
                 cumulative_start=None,
                 cumulative_end=None):
        """
        Plot the distribution of specified data series.

        Parameters:
        -----------
        target_col: str
            Data to plot.

        Returns:
        --------
        
        """
        min_v = self.data[target_col].min()
        max_v = self.data[target_col].max()
        # compute cumulative distribution
        if cumulative_start is None:
            cumulative_start = min_v
        if cumulative_end is None:
            cumulative_end = max_v
        tmp = []
        p = cumulative_start
        while p <= cumulative_end:
            d_t = self.data.loc[self.data[target_col] <= p]
            s = d_t.shape[0]
            tmp.extend([p] * s)
            p += 1
        cumulative = pd.DataFrame({target_col: tmp})

        # plot distribution
        if title is None:
            title = target_col
        fig, axarr = plt.subplots(1, 2, figsize=(15, 5))
        bins = max_v - min_v + 1
        ax = self.data[[target_col]].hist(bins=bins, ax=axarr[0])[0]
        ax.set_xlabel(title)
        ax.set_title('Distribution')
        ax.set_ylabel('Count')

        bins = cumulative_end - cumulative_start + 1
        ax = cumulative[[target_col]].hist(bins=bins, ax=axarr[1])[0]
        ax.set_xlabel(title)
        ax.set_title('Cumulative Distribution')
        ax.set_ylabel('Cumulative')


    def set_feature_info(self, feature_info):
        """
        Set feature information.
        """
        self.feature_info = feature_info


    def set_label_col(self, label_col):
        """
        Set label column.

        label_col: str
            Label column.
        """
        self.label_col = label_col


    def numerical_feature_distplot(self, n_col=3, include=None, exclude=None, bins=20):
        """
        Plot the distrubution of numerical feature.
        """
        if self.feature_info is None:
            print('Feature information missing! See help documents.')
            return
        numerical_features = list(set(self.feature_info.loc[self.feature_info['feature_type'] ==
                                                            'numerical'].index) &
                                  set(self.data.columns.values))
        if include is not None:
            numerical_features = list(set(numerical_features) & set(include))
        if exclude is not None:
            numerical_features = list(set(numerical_features) - set(exclude))
        fig, axarr = DataDescriber._multiplotaxis(n_plot=len(numerical_features),
                                                  n_col=n_col)
        i = 0
        for c in numerical_features:
            try:
                ax = self.data[c].hist(ax=axarr[i],
                                       bins=max(bins, self.data[c].unique().shape[0]))
                ax.set_title(c)
            except:
                print('Error: {}'.format(c))
            i += 1
        plt.draw()
        plt.subplots_adjust(left=0.1, bottom=0,
                            right=0.9, top=0.94,
                            hspace=0.5, wspace=0.3)


    def numerical_feature_boxplot(self, n_col=3,
                                  include=None, exclude=None,
                                  showfliers=False):
        """
        Boxplot for numerical features.
        """
        if self.feature_info is None:
            print('Feature information missing! See help documents.')
            return
        numerical_features = list(set(self.feature_info.loc[self.feature_info['feature_type'] ==
                                                            'numerical'].index) &
                                  set(self.data.columns.values))
        if include is not None:
            numerical_features = list(set(numerical_features) & set(include))
        if exclude is not None:
            numerical_features = list(set(numerical_features) - set(exclude))
        fig, axarr = DataDescriber._multiplotaxis(n_plot=len(numerical_features),
                                                  n_col=n_col)
        i = 0
        for c in numerical_features:
            try:
                ax = sns.boxplot(y=self.data[c], x=self.data[self.label_col],
                                 showfliers=showfliers, ax=axarr[i],
                                 palette="coolwarm")
                ax.set_title(c)
                ax.set_ylabel('')
            except:
                print('Error: {}'.format(c))
            i += 1
        plt.draw()
        plt.subplots_adjust(left=0.1, bottom=0,
                            right=0.9, top=0.94,
                            hspace=0.5, wspace=0.3)


    @staticmethod
    def _distplot(data, target_col, color, title='', xlabel='', xtickinterval=None, logy=False, **kwargs):
        """
        A static helper function to plot
        distribution.
        """
        ax = data[[target_col]].plot(kind='bar', logy=logy, color=color, **kwargs)
        plt.draw()
        if xtickinterval is not None:
            for x in ax.get_xticklabels():
                x.set_visible(False)
            for x in ax.get_xticklabels()[::xtickinterval]:
                x.set_visible(True)
        if logy:
            title += ' (Log)'
        ax.set_title(title)
        ax.grid(axis='y')
        if logy:
            ax.set_ylabel('Count (Log)')
        else:
            ax.set_ylabel('Count')
        ax.set_xlabel(xlabel)
        plt.show()
        return ax


    @staticmethod
    def _multiplotaxis(n_plot, n_col, figsize=(3, 4)):
        """
        A static helper function that creates
        axis for multiple plots.
        """
        a, b = divmod(n_plot, n_col)
        if b == 0:
            n_row = a
        else:
            n_row = a + 1

        horizontal_len, vertical_len = figsize
        fig, axarr = plt.subplots(n_row,
                                  n_col,
                                  figsize=(n_col * vertical_len,
                                           n_row * horizontal_len))
        i = 0
        ret_axis = []
        if (n_row == 1) or (n_col == 1):
            for ax in axarr:
                if i >= n_plot:
                    _ = ax.axis('off')
                    continue
                    ret_axis.append(ax)
                i += 1
        else:
            for subaxarr in axarr:
                for ax in subaxarr:
                    if i >= n_plot:
                        _ = ax.axis('off')
                        continue
                    ret_axis.append(ax)
                    i += 1

        return fig, ret_axis