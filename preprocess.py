from ont_fast5_api.fast5_interface import get_fast5_file
import os
import glob
import click
import torch
import numpy as np
from scipy import stats
import random

def normalization(data_test,filepath):
    mad = stats.median_abs_deviation(data_test, axis=1, scale='normal')
    m = np.median(data_test, axis=1)   
    data_test = ((data_test - np.expand_dims(m,axis=1))*1.0) / (1.4826 * np.expand_dims(mad,axis=1))

    x = np.where(np.abs(data_test) > 3.5)
    for i in range(x[0].shape[0]):
        if x[1][i] == 0:
            data_test[x[0][i],x[1][i]] = data_test[x[0][i],x[1][i]+1]
        elif x[1][i] == 2999:
            data_test[x[0][i],x[1][i]] = data_test[x[0][i],x[1][i]-1]
        else:
            data_test[x[0][i],x[1][i]] = (data_test[x[0][i],x[1][i]-1] + data_test[x[0][i],x[1][i]+1])/2
    
    data_test = torch.from_numpy(data_test.astype(np.float32)).clone()
    torch.save(data_test,filepath)
    
    return data_test
    

class Preprocess():
	### read in pos and neg ground truth variables
    def __init__(self,target) -> None:
        my_file = open(target, "r")
        li = my_file.readlines()
        my_file.close()
        self.li = [pi.split('\n')[0] for pi in li]
        self.filepath = target

        pass
	
    def process(self,inpath,cutoff,length):
        arrpos = []
        filepath = self.filepath.replace('.txt','.pt')
        count = 0
        if (os.path.isfile(filepath)):
            return torch.load(filepath)
        
        else:
            files = glob.glob(inpath + '/*.fast5')
            random.shuffle(files)
            for fileNM in files:
                with get_fast5_file(fileNM, mode="r") as f5:
                    #print("##### file: " + fileNM)
                    if count == length:
                        print(count)
                        break
                    for read in f5.get_reads():
                        raw_data = read.get_raw_data(scale=True)

                        ### only parse reads that are long enough
                        if len(raw_data) >= (cutoff + 3000) and read.read_id in self.li:
                            arrpos.append(raw_data[cutoff:(cutoff + 3000)])
                            count += 1
            return normalization(arrpos,filepath) 