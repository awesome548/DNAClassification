from scipy import stats
import torch
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
FAST5 = os.environ["FAST5"]
MAXLEN = int(os.environ["MAXLEN"])

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

####################
## 
##  PAF FUNCTION
##
####################
def parse_paf(infile,mapdict):
    infile = open(infile)
    past = ""
    for l in infile:
        if not l.startswith("@"):
            tabs = l.split()
            if (tabs[0] != past) and (tabs[2] != "*"):
                mapdict[tabs[0]] = tabs[2].split("_")[0]
                past = tabs[0]

def parse_paf_idx(pafpath,mapdict):
    infile = open(pafpath)
    for l in infile:
        if not l.startswith("@"):
            tabs = l.split()
            if (tabs[2] != "*") and (tabs[1] == "0"):
                mapdict[tabs[2]].append([tabs[0],int(tabs[3]),int(tabs[3])+len(tabs[9])])
                """
                if tabs[1] == "0":
                    MAPPED[tabs[2]] = (tabs[0],tabs[3],tabs[3]+len(tabs[9]))
                elif tabs[1] == "16":
                    MAPPED[tabs[2]] = (tabs[0],tabs[3]-len(tabs[9]),tabs[3])
                past = tabs[0]
                """
    return mapdict

def map_position(mapdict):
    #print(MAPPED)
    overlap_sum = []
    for array in mapdict.values():
        if len(array) != 0:
            pairs = find_overlapping_pairs_optimized(array)
            if pairs != []:
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