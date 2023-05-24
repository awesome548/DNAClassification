import os
from os.path import abspath
import glob
import click
from ops_data.dataformat import Dataformat
from ont_fast5_api.fast5_interface import get_fast5_file
from dotenv import load_dotenv

def main():
	load_dotenv()
	IDLIST = os.environ['IDLIST']
	FAST5 = os.environ['FAST5']
	MISC = os.environ['MISC']
	cutoff = os.environ['CUTOFF']
	maxlen = os.environ['MAXLEN']
	### make output folder
	if not os.path.exists(IDLIST):
		os.makedirs(IDLIST)

	species = []
	input_dir = []
	### fast5 を取得
	if os.path.exists(FAST5):
		for name in glob.glob(FAST5+'/*'):
			dirname = abspath(name)
			if os.path.exists(dirname + '/fast5'):
				species.append(os.path.basename(name))
				input_dir.append(dirname + '/fast5')
	else:
		raise NotImplementedError
	
	print(species)
	print(input_dir)

	### load data
	output = {}
	for spe,input in zip(species,input_dir):
		if (os.path.isfile(IDLIST + '/' + spe + '.txt')) is False:
			ids_list = []
			for fileNM in glob.glob(input + '/*.fast5'):
				with get_fast5_file(fileNM, mode="r") as f5:
					for read in f5.get_reads():
						raw_data = read.get_raw_data(scale=True)
						if len(raw_data) >= (cutoff + maxlen):
								ids_list.append(read.read_id)

			with open(IDLIST + '/' + spe + '.txt', 'a') as f:
				for id in ids_list:
					f.write(str(id)+'\n')
			output[spe] = len(ids_list)
			print("target : " + str(len(ids_list)))
		else:
			raise FileExistsError('text file already exists')
	
	with open(MISC + '/fast5_id_summary.txt','a') as f:
		for i in output.keys():
			f.write(str(i)+'\n')
			f.write('num : '+str(output[i])+'\n')

	"""
	Validation

	maxlen = 10000
	cut_size = {
        'cutoff' : cutoff,
        'cutlen' : cutlen,
        'maxlen' : maxlen,
        'stride' : 5000 if cutlen<=5000 else (10000-cutlen),
    }
	dataset_size = 10000
	data = Dataformat(OUTPUT,FAST5,dataset_size,cut_size,num_classes=6)
	dataset_size = data.size()
	print(dataset_size)
	"""
if __name__ == "__main__":
	main()