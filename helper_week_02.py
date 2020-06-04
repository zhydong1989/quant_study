## This is the helper functions and global variables for this training course

## path of our program
#HEAD_PATH = "/mnt/hgfs/intern"
HEAD_PATH = "d:/intern"


## path of data
DATA_PATH = HEAD_PATH + "/pkl tick/"

## path of the day-to-night data set
NIGHT_PATH = HEAD_PATH + "/night pkl tick/"

## get all the dates of product
import os
def get_dates(product):
    return list(map(lambda x: x[:8] ,os.listdir(DATA_PATH + product)))

## get the data of product of specific date
import pandas as pd
import _pickle as cPickle
import gzip
#import lz4.frame
def get_data(product, date):
    with gzip.open(DATA_PATH + product+"/"+date+".pkl", 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    data = cPickle.loads(raw_data)
    return data


def load(path):
    with gzip.open(path, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    return cPickle.loads(raw_data)

def save(data, path):
    serialized = cPickle.dumps(data)
    with gzip.open(path, 'wb', compresslevel=1) as file_object:
        file_object.write(serialized)
        

## returns 0 if the numerator or denominator is 0
import numpy as np
import warnings
def zero_divide(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = np.divide(x,y)
    if hasattr(y, "__len__"):
        res[y == 0] = 0
    elif y == 0:
        res = 0
        
    return res

def ewma(x, halflife, init=0):
    init_s = pd.Series(data=init)
    s = init_s.append(x)
    return s.ewm(halflife=halflife, adjust=False).mean()[1:]

## moving sum of x
## we don't use rollSum because rollSum would make the first n data to be zero
def cum(x, n):
    sum_x = x.cumsum()
    sum_x_shift = sum_x.shift(n)
    sum_x_shift[:n]= 0
    return sum_x - sum_x_shift


def sharpe(x):
    return zero_divide(np.mean(x)* np.sqrt(250), np.std(x, ddof=1))

def drawdown(x):
    y = np.cumsum(x)
    return np.max(y)-np.max(y[-1:])

def max_drawdown(x):
    y = np.cumsum(x)
    return np.max(np.maximum.accumulate(y)-y)

from collections import OrderedDict
def get_hft_summary(result, thre_mat, n):
    all_result = pd.DataFrame(data={"daily.result": result})
    daily_num = all_result['daily.result'].apply(lambda x: x["num"])
    daily_pnl = all_result['daily.result'].apply(lambda x: x["pnl"])
    total_num = daily_num.sum()
    if len(total_num) != len(thre_mat):
        raise selfException("Mismatch!")
    total_pnl = daily_pnl.sum()
    avg_pnl = zero_divide(total_pnl, total_num)

    total_sharp = sharpe(daily_pnl)
    total_drawdown = drawdown(daily_pnl)
    total_max_drawdown = max_drawdown(daily_pnl)

    final_result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"]), ("close", thre_mat["close"]), ("num", total_num),
                                                 ("avg.pnl", avg_pnl), ("total.pnl", total_pnl), ("sharp", total_sharp), 
                                                 ("drawdown", total_drawdown), ("max.drawdown", total_max_drawdown), 
                                                 ("mar", total_pnl/total_max_drawdown)]), 
                                index=thre_mat.index)
    
    return OrderedDict([("final.result", final_result), ("daily.num", daily_num), ("daily.pnl", daily_pnl)])