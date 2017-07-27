# coding=utf-8
"""
Scripts to encode text address to (lat, lon) coordinates.
File structure must follow:
    ./data/tmp/api_key.txt             --->  BaiduMap API key
    ./data/tmp/loan_no_label.csv       --->  loan no & label
    ./data/tmp/data_eng.csv            --->  text address
    ./data/tmp/tmp_lat_lon_data.csv    --->  processed data
    ./data/tmp/log.txt                --->  log
"""
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
import os
import socket


def encode_addr(amt_limit=10000,
                max_retry=7,
                start_time=None,
                buffer_size=10):
    """
    Encode text address to (lat, lon) coordinates.
    """
    time_started = pd.datetime.now()
    # load api authentic key
    ak_path = './data/tmp/api_key.txt'
    if not os.path.exists(ak_path):
        print('[!].Please add authentic key to {}'.format(ak_path))
        print('[!].The file looks like: "key: [your authentic key]"')
        Return 
    with open('./data/tmp/api_key.txt', 'r') as f:
        content = yaml.load(f)
        ak = content['key']

    # set default timeout to 20 seconds
    socket.setdefaulttimeout(20)

    # label data
    label = pd.read_csv('./data/tmp/loan_no_label.csv',index_col=0)
    # loan no, to check if data is processed
    loan_no = [str(x) for x in label.index]
    label.index = loan_no

    # load text address, and its fields
    text_addr = pd.read_csv('./data/tmp/data_eng.csv',
                            encoding='gb18030',
                            index_col=None,
                            dtype={'loan_no': str})
    text_addr = text_addr.set_index('loan_no')
    addr_fields = text_addr.columns.values[:4]

    # get processed data
    processed = pd.read_csv('./data/tmp/tmp_lat_lon_data.csv',
                            encoding='gb18030',
                            index_col=None,
                            dtype={'loan_no': str})
    processed_loan_no = processed['loan_no'].values

    # request parameters
    url = 'http://api.map.baidu.com/geocoder/v2/'
    # output format
    output = 'json'

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
    text_addr = text_addr.loc[~text_addr.index.isin(processed_loan_no)]
    t1 = time.time()
    output_buffer = ''
    curr_buffer_size = 0
    for curr_loan_no, row in text_addr.iterrows():

        curr_line = curr_loan_no
        for k in addr_fields:
            addr_to_encode = text_addr.loc[curr_loan_no, k]
            add = quote(addr_to_encode)
            uri = url + '?' + 'address=' + add  + \
                  '&output=' + output + '&ak=' + ak

            # request for encoding
            # keep requesting for max_retry times to
            # deal with timeout error
            succeed = False
            for _ in range(max_retry):
                try:
                    req = urlopen(uri)
                    succeed = True
                    break
                except:
                    a = 1

            # if failed, terminate program
            if not succeed:
                if curr_buffer_size > 0:
                    with open('./data/tmp/tmp_lat_lon_data.csv', 'a') as f:
                        f.write(output_buffer)
                m1 ='Request failed: timeout' 
                m = '{}, {}, {}'.format(m1, curr_loan_no, k)
                print(m)
                return

            # format result
            res = req.read().decode()
            res = json.loads(res)
            status = res['status']

            c = '"({}, {})"'
            if res['status'] == 0:
                c = c.format(res['result']['location']['lat'],
                             res['result']['location']['lng'])
            else:
                c = c.format(np.nan, np.nan)

            # append (lat, lon) to current line
            curr_line += ',' + c

        # add label to the end of the line
        curr_line += ',' + str(label.loc[curr_loan_no, 'label']) + '\n'
        output_buffer += curr_line
        curr_buffer_size += 1

        # write lines in the buffer to the end of data file
        if curr_buffer_size >= buffer_size:
            with open('./data/tmp/tmp_lat_lon_data.csv', 'a') as f:
                f.write(output_buffer)
                output_buffer = ''
                curr_buffer_size = 0

        cnt += 1
        pct = round(cnt / amt_limit * 100)
        sys.stdout.write('    {} processed ...    {}%\r'.format(cnt, pct))
        sys.stdout.flush()

        # teminate program if the target is met
        if cnt >= amt_limit:
            break

    time.sleep(1)

    if curr_buffer_size > 0:
        with open('./data/tmp/tmp_lat_lon_data.csv', 'a') as f:
            f.write(output_buffer)

    t2 = time.time()
    time_finished = pd.datetime.now()
    m, s = divmod(t2 - t1, 60)

    # write log
    time_started = 'Started at: {}'.format(time_started)
    time_spent = 'Time spent: {:8.4}min {:6.4}sec'.format(m, s)
    processed_cnt = 'Processed {} row(s)'.format(cnt)
    time_finished = 'Stopped at: {}'.format(time_finished)
    log_info = [time_started, time_spent, processed_cnt, time_finished]

    with open('./data/tmp/log.txt', 'a') as f:
        f.write(time_started + '\n')
        f.write(scheduled_time + '\n')
        f.write(time_spent + '\n')
        f.write(processed_cnt + '\n')
        f.write(time_finished + '\n')
        f.write('\n\n\n')

    print(time_started)
    print(scheduled_time)
    print(time_spent)
    print(processed_cnt)
    print(time_finished)
    print()


def main():
    """
    Handle command line options.
    """
    parser = argparse.ArgumentParser()

    # commond
    # parser.add_argument('-', '--help', default=False,
    #                 help='Display help information')

    parser.add_argument('-t', '--start_time', default=None,
                        help='Start time')

    parser.add_argument('-a', '--amt', default=10000, type=int,
                        help='Number of row to process')

    parser.add_argument('-r', '--retry', default=7, type=int,
                        help='Number of times to retry after timeout')

    parser.add_argument('-b', '--buffer_size', default=10, type=int,
                        help='Output buffer size')

    args = parser.parse_args()

    # get data
    encode_addr(amt_limit=args.amt,
                max_retry=args.retry,
                start_time=args.start_time,
                buffer_size=args.buffer_size)


if __name__ =='__main__':
    main()
