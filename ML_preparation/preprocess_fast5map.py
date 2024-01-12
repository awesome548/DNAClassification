import os
from collections import defaultdict
from dotenv import load_dotenv
import csv
import glob
import numpy as np
from utils import mad_normalization,manipulate
from ont_fast5_api.fast5_interface import get_fast5_file
from utils import parse_paf_idx,map_position
from concurrent.futures import ProcessPoolExecutor
load_dotenv()
MISC = os.environ["MISC"]
CUTOFF = int(os.environ['CUTOFF'])
MAXLEN = int(os.environ['MAXLEN'])
DATASIZE = int(os.environ["DATASETSIZE"])
DATAPATH = os.environ['DATAPATH']
ZYMOPATH = os.environ['ZYMO']

def process_file(fileNM, id_set, CUTOFF, MAXLEN):
    arrpos = []
    with get_fast5_file(fileNM, mode="r") as f5:
        for read in f5.get_reads():
            if read.read_id in id_set:
                print("success")
                raw_data = read.get_raw_data()
                if len(raw_data) >= (CUTOFF + MAXLEN):
                    arrpos.append(raw_data[CUTOFF:(CUTOFF + MAXLEN)])
    return arrpos

def main():
    pafpath = "/z/nanopore/zymo_alignment.paf"
    mapdict = defaultdict(list)
    ### Loading PAF File
    print("loading paf....")
    mapdict = parse_paf_idx(pafpath,mapdict)
    print("paf loaded!!")
    result = map_position(mapdict)

    with open(f'{MISC}/output.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(result)
    
    #mostpaired_contig = max(result, key=len)
    target_contig  = result[35]

    id_list = []
    for e in target_contig:
        id_list.append(e[0][0])
        id_list.append(e[1][0])
    #print(id_list)
    print(len(id_list))
    print(id_list[0])

    id_list = set(id_list)
    files = glob.glob(f'{ZYMOPATH}/batch_*.fast5')
    arrpos = process_file(files,id_list,CUTOFF,MAXLEN)
    
    print(f'processed num of fast5 : {len(arrpos)}')
    x = mad_normalization(np.array(arrpos), f'{DATAPATH}/noise_test.pt')
    return manipulate(x)


if __name__ == "__main__":
    main()