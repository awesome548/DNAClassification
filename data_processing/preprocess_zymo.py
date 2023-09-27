import os
import torch
from scipy import stats
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
import argparse
import glob
from preprocess import mad_normalization
from dotenv import load_dotenv

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

    def parse_paf(infile):
        infile = open(infile)
        past = ""
        for l in infile:
            if not l.startswith("@"):
                tabs = l.split()
                if (tabs[0] != past) and (tabs[2] != "*"):
                    MAPPED[tabs[0]] = tabs[2].split("_")[0]
                    past = tabs[0]

    ### Loading PAF File
    print("loading paf....")
    parse_paf(PAF)
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