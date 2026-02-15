from scipy import stats
import torch
import numpy as np
import os
from dotenv import load_dotenv
from ML_preparation.plot import plot_torch_1d

load_dotenv()
FAST5 = os.environ["FAST5"]
DATASIZE = int(os.environ["DATASETSIZE"])
CUTOFF = int(os.environ["CUTOFF"])
MAXLEN = int(os.environ["MAXLEN"])
CUTLEN = int(os.environ["CUTLEN"])
DATAPATH = os.environ['DATAPATH']
STRIDE = int(os.environ["STRIDE"])

####################
## 
##  FAST5 FUNCTION
##
####################
def mad_normalization(np_array,output_filepath):
    mad = stats.median_abs_deviation(np_array, axis=1, scale='normal')
    m = np.median(np_array, axis=1)   
    base_data = ((np_array - np.expand_dims(m,axis=1))*1.0) / (1.4826 * np.expand_dims(mad,axis=1))
    x = np.where(np.abs(base_data) > 3.5)
    for i in range(x[0].shape[0]):
        if x[1][i] == 0:
            base_data[x[0][i],x[1][i]] = base_data[x[0][i],x[1][i]+1]
        elif x[1][i] == (MAXLEN-1):
            base_data[x[0][i],x[1][i]] = base_data[x[0][i],x[1][i]-1]
        else:
            base_data[x[0][i],x[1][i]] = (base_data[x[0][i],x[1][i]-1] + base_data[x[0][i],x[1][i]+1])/2

    norm_data = torch.from_numpy(base_data.astype(np.float32)).clone()
    print(f'saved torch file size : {norm_data.shape}')
    torch.save(norm_data,output_filepath)
    print(f'file saved to : {output_filepath}')
    return norm_data

def calu_ratio():
    return (MAXLEN - CUTLEN)//STRIDE + 1 

def calu_size():
    return ((MAXLEN - CUTLEN)//STRIDE + 1)*DATASIZE

def manipulate(x):
    num = calu_ratio()
    data = torch.zeros(DATASIZE*num,CUTLEN)
    ## start point is already cutoff point ###
    for index in range(num):
        start = STRIDE*index
        data[index::num,:] = x[:DATASIZE,start:start+CUTLEN]
    print(f'shaped torch size : {data.shape}')
    #plot_torch_1d(data[0,0:1000],0)
    return data
    """
    データ正規化確認
    print(torch.max(data))
    print(torch.min(data))
    data = data.cpu().detach().numpy().copy()
    data = stats.zscore(data,axis=1,ddof=1)
    print(np.max(data))
    print(np.min(data))
    data = torch.from_numpy(data.astype(np.int32)).clone()
    """
####################
## 
##  PAF FUNCTION
##
####################
def parse_paf(infile_path, mapdict):
    past = ""
    with open(infile_path) as infile:
        for line in infile:
            if not line.startswith("@"):
                tabs = line.split()
                if (tabs[0] != past) and (tabs[2] != "*"):
                    mapdict[tabs[0]] = tabs[2].split("_")[0]
                    past = tabs[0]


def parse_paf_idx(pafpath, mapdict):
    with open(pafpath) as infile:
        for line in infile:
            if not line.startswith("@"):
                tabs = line.split()
                if (tabs[2] != "*") and (tabs[1] == "0"):
                    mapdict[tabs[2]].append([tabs[0], int(tabs[3]), int(tabs[3]) + len(tabs[9])])
    return mapdict


def map_position(mapdict):
    overlap_sum = []
    for array in mapdict.values():
        if len(array) != 0:
            pairs = find_overlapping_pairs_optimized(array)
            if pairs:
                overlap_sum.append(pairs)
    return overlap_sum


def find_overlapping_pairs_optimized(array):
    seen = {}
    overlapping_pairs = []

    for sub_array in array:
        key = (sub_array[1],sub_array[2])  # タプルをキーとして使用
        if key in seen:
            overlapping_pairs.append((seen[key], sub_array))
        else:
            seen[key] = sub_array
    return overlapping_pairs
