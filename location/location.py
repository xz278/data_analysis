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
from os.path import join, exists


def get_coordinates(addr_col,
                    df_path='./data/data_eng.csv',
                    result_path='./data/',
                    amt=10000,
                    start_time=None):
    """
    Convert address to gps coordinates.
    Now can only be used in the specified file structure.

    Parameters:
    -----------
    df_path: str
        Address data file path.
        Defaults to './data/data_eng.csv'

    addr_col: str
        Address column.

    result_path: str
        File to store lat,lon results.
        Defualts to './data/'.

    amt: int
        Number of data to process.
        Defaults to 10000.

    start_time: str
        Start time of the process, '2017-6-5 20:30:00'
        Defualt value is None, meanning start the process
        immediately.
    """
    # load data
    time_started = pd.datetime.now()
    data = pd.read_csv(df_path, encoding='gb18030', index_col='loan_no')
    loan_no = [str(x) for x in data.index.values]
    data.index = loan_no

    # load current data
    f_output = join(result_path, '{}.yaml'.format(addr_col))
    if exists(f_output):
        with open(f_output, 'r') as f:
            tmp = yaml.load(f)
    else:
        tmp = {}

    # start the process at the specified time
    if start_time is not None:
        start_time = pd.to_datetime(start_time)
        current_time = pd.datetime.now()
        time_diff = (start_time - current_time).total_seconds()
        if time_diff > 10:
            scheduled_time = 'Scheduled start time: {}'.format(start_time)
            print('Will start at {}\n'.format(start_time))
            time.sleep(time_diff)
        else:
            scheduled_time = 'Scheduled start time: {}'.format('N/A')
    else:
        scheduled_time = 'Scheduled start time: {}'.format('N/A')

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

        # save results after every 100 request
        c_a, c_b = divmod(cnt, 100)
        if c_b == 0:
            with open(f_output, 'w') as f:
                yaml.dump(tmp, f)

        sys.stdout.write('    {} processed ...\r'.format(cnt))
        sys.stdout.flush()
        if cnt >= amt:
            break
        time.sleep(0.1)

    t2 = time.time()
    m, s = divmod(t2 - t1, 60)

    with open(f_output, 'w') as f:
        yaml.dump(tmp, f)

    addr_type = 'Address type: {}'.format(addr_col)
    time_started = 'Started at: {}'.format(time_started)
    time_spent = 'Time spent: {:8.4}min {:6.4}sec'.format(m, s)
    processed = 'Processed {} addresses'.format(cnt)
    time_finished = pd.datetime.now()
    time_finished = 'Stopped at: {}'.format(time_finished)
    log_info = [addr_type, time_started, time_spent, processed, time_finished]
    with open('./data/log.txt', 'a') as f:
        f.write(addr_type + '\n')
        f.write(time_started + '\n')
        f.write(scheduled_time + '\n')
        f.write(time_spent + '\n')
        f.write(processed + '\n')
        f.write(time_finished + '\n')
        f.write('\n\n\n')
    print(addr_type)
    print(time_started)
    print(scheduled_time)
    print(time_spent)
    print(processed)
    print(time_finished)
    print()


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
