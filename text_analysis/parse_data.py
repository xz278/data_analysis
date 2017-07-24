# coding=utf-8
"""
Scripts for to parse and prepare
text input, ouput, response, and keywords
for analysis.

The directory must stikc to the following structure:
    ./
    ./data/
All relavent data must be stored in the 'data' directory,
and all proccessed data will be stored in the 'data' dir.
"""
import numpy as np
import pandas as pd
import utils
# import gc
# import json
# import yaml

    
def preprocess():
    """
    Preprocess data for text analysis.
    """
    # load columns to add
    utf8 = 'utf-8'
    ansi = 'gb18030'
    try:
        cols = list(pd.read_table('./cols.txt', sep=',', encoding=utf8, header=None, index_col=None).iloc[0, :].values)
    except:
        cols = list(pd.read_table('./cols.txt', sep=',', encoding=ansi, header=None, index_col=None).iloc[0, :].values)

    # input data
    input_text = pd.read_csv('./data/input.txt', encoding='gb18030', header=None, index_col=None, sep='\t')
    loan_no = pd.read_csv('./data/loan_number.txt', encoding='gb18030', index_col=None, header=None)
    input_text['loan_no'] = [str(x) for x in loan_no.iloc[:, 0]]
    if '贷款编号' not in cols:
        cols.append('贷款编号')
    input_text.columns = cols
    input_text = input_text.set_index('贷款编号', drop=False)
    input_text['日期'] = utils.get_date_from_loan_no(loan_no_list=input_text['贷款编号'])
    input_text.to_csv('./data/input_with_header_index.csv', encoding='gb18030')

    # output data
    output_text = pd.read_csv('./data/output.txt', header=None, index_col=None, sep=' ', encoding='gb18030')
    key_word = pd.read_csv('./data/keyword.txt', sep=' ', header=None, index_col=None, encoding='gb18030')
    key_word = key_word.dropna(axis=1).iloc[0, :].values
    output_text.columns = key_word
    output_text['贷款编号'] = [str(x) for x in loan_no.iloc[:, 0]]
    output_text = output_text.set_index('贷款编号')
    output_text.to_csv('./data/output_with_header_index.csv', encoding='gb18030')

    # response
    response = pd.read_csv('./data/respond.txt', encoding='gb18030', index_col=None, header=None)
    response['贷款编号'] = [str(x) for x in loan_no.iloc[:, 0]]
    response = response.set_index('贷款编号')
    response.columns = ['label']
    response.to_csv('./data/respond_with_header_index.csv', encoding='gb18030')

    # all data
    data = output_text.join(response)
    data.to_csv('./data/model_data.csv', encoding='gb18030')


def main():
    """
    Functions for command line options.
    """
    preprocess()


if __name__ == '__main__':
    main()