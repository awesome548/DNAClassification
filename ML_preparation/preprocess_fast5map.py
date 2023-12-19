import os
from collections import defaultdict
from dotenv import load_dotenv
import csv
from ML_preparation.utils import parse_paf_idx,map_position
load_dotenv()
MISC = os.environ["MISC"]
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

if __name__ == "__main__":
    main()