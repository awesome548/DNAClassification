import os
import glob
import click
from ops_data.dataformat import Dataformat
from ont_fast5_api.fast5_interface import get_fast5_file

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
@click.option('--idpath', '-id', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--inpath', '-in', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--cutlen', '-len', default=4000, help='Cutting length')
@click.option('--cutoff', '-off', default=1500, help='Cutting length')
@click.option('--classname', '-c', help='The output file name', type=click.Path())
@click.option('--maxlen', '-m', default=20000, help='Cutoff the first c signals')

def main(idpath,inpath,cutlen,cutoff,classname,maxlen):
	### make output folder
	if not os.path.exists(idpath):
		os.makedirs(idpath)

	### load data
	target_list = []
	if (os.path.isfile(idpath + '/' + classname + '.txt')) is False:
		print("making ID lists")
		tardir = inpath+'/'+ classname
		print(tardir)
		for fileNM in glob.glob(tardir + '/*.fast5'):
			#print(fileNM)
			target_list = get_raw_data(fileNM, target_list, cutoff,maxlen)

		num_tar = 0
		with open(idpath + '/' + classname + '.txt', 'a') as f:
			for cont in target_list:
				num_tar +=1
				#print(cont)
				f.write(str(cont)+'\n')

		print("target : " + str(num_tar))

	maxlen = 10000
	cut_size = {
        'cutoff' : cutoff,
        'cutlen' : cutlen,
        'maxlen' : maxlen,
        'stride' : 5000 if cutlen<=5000 else (10000-cutlen),
    }
	dataset_size = 10000
	data = Dataformat(idpath,inpath,dataset_size,cut_size,num_classes=6)
	dataset_size = data.size()
	print(dataset_size)
if __name__ == "__main__":
	main()