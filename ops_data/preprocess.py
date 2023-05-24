import os
import glob
import torch
import random
import numpy as np
from scipy import stats
from ont_fast5_api.fast5_interface import get_fast5_file
from dotenv import load_dotenv

def mad_normalization(data_test,filepath,maxlen):
    mad = stats.median_abs_deviation(data_test, axis=1, scale='normal')
    m = np.median(data_test, axis=1)   
    data_test = ((data_test - np.expand_dims(m,axis=1))*1.0) / (1.4826 * np.expand_dims(mad,axis=1))
    x = np.where(np.abs(data_test) > 3.5)
    for i in range(x[0].shape[0]):
        if x[1][i] == 0:
            data_test[x[0][i],x[1][i]] = data_test[x[0][i],x[1][i]+1]
        elif x[1][i] == (maxlen-1):
            data_test[x[0][i],x[1][i]] = data_test[x[0][i],x[1][i]-1]
        else:
            data_test[x[0][i],x[1][i]] = (data_test[x[0][i],x[1][i]-1] + data_test[x[0][i],x[1][i]+1])/2
    data_test = torch.from_numpy(data_test.astype(np.float32)).clone()
    print(f'saved torch file size : {data_test.shape}')
    torch.save(data_test,filepath)
    print(f'file saved to : {filepath}')
    return data_test

    

def manipulate(x,cutlen,maxlen,size,stride):
    num = calu_size(cutlen,maxlen,stride)
    data = torch.zeros(size*num,cutlen)
    ### start point is already cutoff point ###
    for index in range(num):
        start = stride*index
        data[index::num,:] = x[:size,start:start+cutlen]
    print(f'shaped torch size : {data.shape}')
    return data
    """
    print(torch.max(data))
    print(torch.min(data))
    data = data.cpu().detach().numpy().copy()
    data = stats.zscore(data,axis=1,ddof=1)
    print(np.max(data))
    print(np.min(data))
    data = torch.from_numpy(data.astype(np.int32)).clone()
    """

def calu_size(cutlen,maxlen,stride):
    return (maxlen - cutlen)//stride + 1 


class Preprocess():
	### read in pos and neg ground truth variables
    def __init__(self,id,fast5) -> None:
        load_dotenv()
        DATA = os.environ['DATA']
        if os.path.exists(id) and os.path.exists(fast5):
            my_file = open(id, "r")
            # id を読み込む
            li = my_file.readlines()
            my_file.close()
            self.li = [pi.split('\n')[0] for pi in li]
            self.idpath = id 
            self.fast5path = fast5
            self.outpath = DATA +'/'+ os.path.basename(self.idpath.replace('.txt','.pt'))
            print('##############')
            print(f'input id path : {id}')
            print(f'num of fast5-id : {len(self.li)}')
        else:
            raise FileNotFoundError
        pass

    def __len__(self) -> int:
        return len(self.li)
	
    def process(self,cutoff,cutlen,maxlen,req_size,stride):
        output_path = self.outpath
        f5_path = self.fast5path
        file_exist = False
        count = 0
        print(f5_path+" processing...")
        if (os.path.isfile(output_path)):
            x = torch.load(output_path)
            count = x.shape[0]
            print(f'processed num of fast5 : {count}')
            file_exist = True

        else:
            arrpos = []
            for fileNM in glob.glob(f5_path + '/*.fast5'):
                with get_fast5_file(fileNM, mode="r") as f5:
                    for read in f5.get_reads():
                        raw_data = read.get_raw_data()
                        if len(raw_data) >= (cutoff + maxlen):
                            arrpos.append(raw_data[cutoff:(cutoff + maxlen)])
            print(f'processed num of fast5 : {len(arrpos)}')
        if file_exist:
            pass
        elif len(arrpos) >= req_size:
            x = mad_normalization(np.array(arrpos),output_path,maxlen) 
        else:
            raise IndexError('dataset size is larger than num of fast5 having enough length')
        
        return manipulate(x,cutlen,maxlen,req_size,stride)
    
    