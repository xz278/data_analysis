# coding=utf-8
import numpy as py
import pandas as pd
from utils import get_latlon
from utils import batch_latlon
from urllib.request import urlopen, quote
import requests
import json
import time
import numpy as np
import sys
import yaml
import utils
import argparse


def get_coordinates(addr_col,
                    df_path='./data/locations_with_loan_no_and_label.csv',
                    f_output='./data/tmp.yaml',
                    amt=10000,
                    start_time=None):
    """
    Convert address to gps coordinates.
    Now can only be used in the specified file structure.

    Parameters:
    -----------
    df_path: str
        Address data file path.
        Defaults to './data/locations_with_loan_no_and_label.csv'

    addr_col: str
        Address column.

    f_output: str
        File to store data.
        Defualts to './data/tmp.yaml'.

    amt: int
        Number of data to process.
        Defaults to 10000.

    start_time: str
        Start time of the process, '2017-6-5 20:30:00'
        Defualt value is None, meanning start the process
        immediately.
    """
    # load data
    data = pd.read_csv(df_path, encoding='gb18030', index_col='贷款编号')
    loan_no = [str(x) for x in data.index.values]
    data.index = loan_no

    # load current data
    with open(f_output, 'r') as f:
        tmp = yaml.load(f)

    # start the process at the specified time
    if start_time is not None:
        start_time = pd.to_datetime(start_time)
        current_time = pd.datetime.now()
        if start_time > current_time:
            time_diff = (start_time - current_time).total_seconds()
            if time_diff > 0:
                time.sleep(time_diff)

    cnt = 0
    t1 = time.time()
    for l in loan_no:
        if l in tmp:
            continue
        cnt += 1
        r = get_latlon(text_address=data.loc[l, addr_col])
        if r['succeed']:
            g = (r['lat'], r['lon'])
        else:
            g = (np.nan, np.nan)
        tmp[l] = g
        if cnt >= amt:
            break
        time.sleep(1)

    t2 = time.time()
    m, s = divmod(t2 - t1, 60)

    with open(f_output, 'w') as f:
        yaml.dump(tmp, f)

    time_spent = 'Time spent: {:8.4}min {:6.4}sec'.format(m, s)
    processed = 'Processed {} addresses'.format(cnt)
    log_info = [time_spent, processed]
    with open('./data/log.yaml', 'w') as f:
        yaml.dump(log_info, f)
    print(time_spent)
    print(processed)


def main():
    """
    Handle command line options.
    """



    parser = argparse.ArgumentParser()

    # commond
    parser.add_argument('-ac', '--addr_col', required=True,
                        help='Address column')

    parser.add_argument('-t', '--start_time', default=None,
                        help='Start time')

    parser.add_argument('-a', '--amt', default=10000, type=int,
                        help='Maxium amout of data to process')


    args = parser.parse_args()

    # get data
    get_coordinates(addr_col=args.addr_col,
                    amt=args.amt,
                    start_time=args.start_time)


if __name__ =='__main__':
    main()
