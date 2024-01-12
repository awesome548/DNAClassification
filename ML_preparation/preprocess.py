import os
import glob
import torch
import random
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
from ML_preparation.utils import mad_normalization,manipulate
from dotenv import load_dotenv
load_dotenv()
FAST5 = os.environ["FAST5"]
DATASIZE = int(os.environ["DATASETSIZE"])
CUTOFF = int(os.environ["CUTOFF"])
MAXLEN = int(os.environ["MAXLEN"])
CUTLEN = int(os.environ["CUTLEN"])
DATAPATH = os.environ['DATAPATH']
STRIDE = int(os.environ["STRIDE"])

class Preprocess():
    def __init__(self,fast5,out,flag) -> None:
        self.fast5path = fast5
        self.outpath = DATAPATH +'/'+ out
        if flag:
            if not os.path.exists(fast5):
                raise FileNotFoundError
	
    def process(self):
        f5_path = self.fast5path
        file_exist = False
        print(f5_path+" processing...")
        files = (glob.glob(self.outpath+'*'))
        if files:
            ## もしも同じファイルで再現性を持たせたほうがいい場合はコメントアウトする
            random.shuffle(files)
            x = torch.load(files[0])
            i = 1
            while(x.shape[0] < DATASIZE):
                y = torch.load(files[i])
                x = torch.cat([x,y],dim=0)
                i += 1
            print(f'processed num of fast5 : {x.shape[0]}')
            file_exist = True
        else:
            arrpos = []
            for fileNM in glob.glob(f5_path + '/*.fast5'):
                with get_fast5_file(fileNM, mode="r") as f5:
                    for read in f5.get_reads():
                        raw_data = read.get_raw_data()
                        if len(raw_data) >= (CUTOFF + MAXLEN):
                            arrpos.append(raw_data[CUTOFF:(CUTOFF + MAXLEN)])
            print(f'processed num of fast5 : {len(arrpos)}')
        if file_exist:
            pass
        elif len(arrpos) >= DATASIZE:
            x = mad_normalization(np.array(arrpos),self.outpath) 
        else:
            raise IndexError('dataset size is larger than num of fast5 having enough length')
        
        return manipulate(x)
    
    