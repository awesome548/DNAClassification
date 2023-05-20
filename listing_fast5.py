import os
from os.path import abspath
import glob
import click
from ops_data.dataformat import Dataformat
from ont_fast5_api.fast5_interface import get_fast5_file

OUTPUT = "/z/kiku/Basecaller/DNAClassification/Id_list"
FAST5 = "/z/kiku/Agn_Tar"
MISC = "/z/kiku/Basecaller/DNAClassfication/misc"
# Args setting


########################
#### Load the data #####
########################
def get_raw_data(fileNM, PNlist, cutoff,maxlen):
	with get_fast5_file(fileNM, mode="r") as f5:
		for read in f5.get_reads():
			raw_data = read.get_raw_data(scale=True)
			if len(raw_data) >= (cutoff + maxlen):
					PNlist.append(read.read_id)
	return PNlist

@click.command()
@click.option('--cutlen', '-len', default=4000, help='Cutting length')
@click.option('--cutoff', '-off', default=1500, help='Cutting length')
@click.option('--maxlen', '-m', default=20000, help='Cutoff the first c signals')
def main(cutlen,cutoff,maxlen):
	### make output folder
	if not os.path.exists(OUTPUT):
		os.makedirs(OUTPUT)

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
	target_list = []
	output = {}
	for spe,input in zip(species,input_dir):
		if (os.path.isfile(OUTPUT + '/' + spe + '.txt')) is False:
			for fileNM in glob.glob(input + '/*.fast5'):
				target_list = get_raw_data(fileNM, target_list, cutoff,maxlen)

			num_tar = 0
			with open(OUTPUT + '/' + spe + '.txt', 'a') as f:
				for cont in target_list:
					num_tar +=1
					f.write(str(cont)+'\n')
			output[spe] = num_tar
			print("target : " + str(num_tar))
		else:
			raise FileExistsError('text file already exists')
	
	with open(MISC + '/fast5_id_summary.txt','a') as f:
		for i in output.keys:
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