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
    data = load(DATA_PATH + product+"/"+date+".pkl")
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


from collections import OrderedDict
def get_signal_pnl(file, product, signal_name, thre_mat, reverse=1, tranct=1.1e-4, HEAD_PATH="d:/intern"):
    ## load data
    data = load(HEAD_PATH+"/night pkl tick/"+product+"/"+file)
    data = data[data["good"]]
    #n_bar = len(data)
    
    ## load signal
    S = load(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file)

    ## we don't know the signal is positive correlated or negative correlated  
    pred = S*reverse
    #n_thre = len(thre_mat)
    result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"].values), ("close", thre_mat["close"].values),
                               ("num", 0), ("avg.pnl", 0), ("pnl", 0)]), index=thre_mat.index)
    
    for thre in thre_mat.iterrows():
        buy = pred>thre[1]["open"]
        sell = pred<-thre[1]["open"]
        signal = pd.Series(data=0, index=data.index)
        position = signal.copy()
        signal[buy] = 1
        signal[sell] = -1
        scratch = -thre[1]["close"]
        position_pos = pd.Series(data=np.nan, index=data.index)
        position_pos.iloc[0] = 0
        position_pos[(signal==1) & (data["next.ask"]>0) & (data["next.bid"]>0)] = 1
        position_pos[(pred< -scratch) & (data["next.bid"]>0)] = 0
        position_pos.ffill(inplace=True)
        position_neg = pd.Series(data=np.nan, index=data.index)
        position_neg.iloc[0] = 0
        position_neg[(signal==-1) & (data["next.ask"]>0) & (data["next.bid"]>0)] = -1
        position_neg[(pred> scratch) & (data["next.ask"]>0)] = 0
        position_neg.ffill(inplace=True)
        position = position_pos + position_neg
        #position[n_bar-1] = 0
        position.iloc[0] = 0
        position.iloc[-2:] = 0
        change_pos = position - position.shift(1)
        change_pos.iloc[0] = 0
        change_base = pd.Series(data=0, index=data.index)
        change_buy = change_pos>0
        change_sell = change_pos<0
        change_base[change_buy] = (data["next.ask"][change_buy])*(1+tranct)
        change_base[change_sell] = (data["next.bid"][change_sell])*(1-tranct)
        final_pnl = -sum(change_base*change_pos)

        num = sum((position!=0) & (change_pos!=0))
        
        if num == 0:
            avg_pnl = 0
            final_pnl = 0
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)
            return result
        else:
            avg_pnl = np.divide(final_pnl, num)
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)
            
    return result


def vanish_thre(x, thre):
    x[np.abs(x)>thre] = 0
    return x


import itertools
def create_signal_path(signal_list, product, HEAD_PATH):
    keys = list(signal_list.params.keys())
    
    for cartesian in itertools.product(*signal_list.params.values()):
        signal_name = signal_list.factor_name
        for i in range(len(cartesian)):
            signal_name = signal_name.replace(keys[i], str(cartesian[i]))
        
        os.makedirs(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name, exist_ok=True)
        
        

def get_signal_pnl_better(file, product, signal_name, thre_mat, reverse=1, tranct=1.1e-4, min_spread=1.1, HEAD_PATH="d:/intern"):
    ## load data
    data = load(HEAD_PATH+"/night pkl tick/"+product+"/"+file)
    data = data[data["good"]]
    #n_bar = len(data)
    
    ## load signal
    S = load(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file)
    
    ## we don't know the signal is positive correlated or negative correlated  
    pred = S*reverse
    #n_thre = len(thre_mat)
    result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"].values), ("close", thre_mat["close"].values),
                               ("num", 0), ("avg.pnl", 0), ("pnl", 0)]), index=thre_mat.index)
    
    bid_ask_spread = data["ask"]-data["bid"]
    next_spread = bid_ask_spread.shift(-1)
    next_spread.iloc[-1] = bid_ask_spread.iloc[-1]
    not_trade = (data["time"]=="10:15:00") | (data["time"]=="11:30:00") | (data["time"]=="15:00:00") | (bid_ask_spread>min_spread) | (next_spread>min_spread)

    for thre in thre_mat.iterrows():
        buy = pred>thre[1]["open"]
        sell = pred<-thre[1]["open"]
        signal = pd.Series(data=0, index=data.index)
        position = signal.copy()
        signal[buy] = 1
        signal[sell] = -1
        signal[not_trade] = 0
        scratch = -thre[1]["close"]
        position_pos = pd.Series(data=np.nan, index=data.index)
        position_pos.iloc[0] = 0
        position_pos[(signal==1) & (data["next.ask"]>0) & (data["next.bid"]>0)] = 1
        position_pos[(pred< -scratch) & (data["next.bid"]>0)] = 0
        position_pos.ffill(inplace=True)
        position_neg = pd.Series(data=np.nan, index=data.index)
        position_neg.iloc[0] = 0
        position_neg[(signal==-1) & (data["next.ask"]>0) & (data["next.bid"]>0)] = -1
        position_neg[(pred> scratch) & (data["next.ask"]>0)] = 0
        position_neg.ffill(inplace=True)
        position = position_pos + position_neg
        #position[n_bar-1] = 0
        position.iloc[0] = 0
        position.iloc[-2:] = 0
        change_pos = position - position.shift(1)
        change_pos.iloc[0] = 0
        change_base = pd.Series(data=0, index=data.index)
        change_buy = change_pos>0
        change_sell = change_pos<0
        change_base[change_buy] = (data["next.ask"][change_buy])*(1+tranct)
        change_base[change_sell] = (data["next.bid"][change_sell])*(1-tranct)
        final_pnl = -sum(change_base*change_pos)

        num = sum((position!=0) & (change_pos!=0))
        
        if num == 0:
            avg_pnl = 0
            final_pnl = 0
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)
            return result
        else:
            avg_pnl = np.divide(final_pnl, num)
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)
            
    return result




def get_range_pos(wpr, min_period, max_period, period):
    return ewma(zero_divide(wpr-min_period, max_period-min_period), period) - 0.5


import dask
from dask import compute, delayed
import matplotlib.pyplot as plt
import functools
from collections import OrderedDict
def get_signal_stat_better(signal_name, thre_mat, product, good_night_files, split_str="2018", reverse=1, min_pnl=2, min_spread=1,CORE_NUM=4):
    train_sample = good_night_files<split_str
    test_sample = good_night_files>split_str
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl_better, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse,
                                 min_spread=min_spread)
        train_result = compute([delayed(f_par)(file) for file in good_night_files[train_sample]])[0]
    train_stat = get_hft_summary(train_result, thre_mat, sum(train_sample))
    good_strat = train_stat["final.result"]["avg.pnl"]>min_pnl
    print("good strategies: \n", good_strat[good_strat], "\n")
    
    good_pnl = train_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
    
    print("train sharpe: ", sharpe(good_pnl), "\n")
    
    date_str = [n[0:8] for n in good_night_files]
    format_dates = np.array([pd.to_datetime(d) for d in date_str])
    plt.figure(1, figsize=(16, 10))
    plt.title("train")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[train_sample], good_pnl.cumsum())
    
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl_better, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse,
                                 min_spread=min_spread)
        test_result = compute([delayed(f_par)(file) for file in good_night_files[test_sample]])[0]
    test_stat = get_hft_summary(test_result, thre_mat, sum(test_sample))
    
    test_pnl = test_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
    
    print("test sharpe: ", sharpe(test_pnl), "\n")
    
    plt.figure(2, figsize=(16, 10))
    plt.title("test")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[test_sample], test_pnl.cumsum())
    
    return OrderedDict([("train.stat", train_stat), ("test.stat", test_stat), ("good.strat", good_strat)])



def rsi(ret, period):
    abs_move = np.abs(ret)
    up_move = np.maximum(ret, 0)
    up_total = ewma(up_move, period)
    move_total = ewma(abs_move, period)
    rsi = zero_divide(up_total, move_total) - 0.5
    return rsi