import os
import torch
from scipy import stats
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
import glob
from dotenv import load_dotenv
from utils import parse_paf,mad_normalization

def main():
    load_dotenv()
    MAPPED = {}
    RAWDATA = {
        "pa" : [],
        "ec" : [],
        "se" : [],
        "lf" : [],
        "ef" : [],
        "sa" : [],
        "lm" : [],
        "bs" : [],
        "sc" : [],
        "cn" : [],
    }
    SPECIES = {
        "pa" : "Pseudomonas_aeruginosa",
        "ec" : "Escherichia_coli",
        "se" : "Salmonella_enterica",
        "lf" : "Lactobacillus_fermentum",
        "ef" : "Enterococcus_faecalis",
        "sa" : "Staphylococcus_aureus",
        "lm" : "Listeria_monocytogenes",
        "bs" : "Bacillus_subtilis",
        "sc" : "Saccharomyces_cerevisiae",
        "cn" : "Cryptococcus_neoformans",
    }
    PAF = "/z/nanopore/zymo_alignment.paf"
    cutoff = int(os.environ['CUTOFF'])
    DATA = os.environ['DATA']
    maxlen = int(os.environ['MAXLEN'])


    ### Loading PAF File
    print("loading paf....")
    MAPPED = parse_paf(PAF,MAPPED)
    print("paf loaded!!")
    count = 0
    for fileNM in glob.glob('/z/nanopore/Zymo-GridION-EVEN-BB-SN-PCR-R10HC_multi/batch_*.fast5'):
        print(f'{fileNM} loading...')
        count += 1
        if count%2 != 0:
            with get_fast5_file(fileNM,mode="r") as f5:
                for read in f5.get_reads():
                    raw_data = read.get_raw_data()
                    if len(raw_data) >= (cutoff + maxlen):
                        try:
                            spe = MAPPED[read.read_id]
                            RAWDATA[spe].append(raw_data[cutoff:(cutoff + maxlen)])
                        except KeyError:
                            pass
        else:
            print("saveing to torch ...")
            for i in RAWDATA.keys():
                if len(RAWDATA[i]) > 10000:
                    _ = mad_normalization(np.array(RAWDATA[i]),DATA+'/'+SPECIES[i]+str(count)+'.pt',maxlen)
                    RAWDATA[i] = [] 

if __name__ == "__main__":
    main()