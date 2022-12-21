import click
import os
import glob
from ont_fast5_api.fast5_interface import get_fast5_file


########################
#### Load the data #####
########################
def get_raw_data(infile, fileNM, PNlist, cutoff,cutlen):
	fast5_filepath = os.path.join(infile, fileNM)
	with get_fast5_file(fast5_filepath, mode="r") as f5:
		for read in f5.get_reads():
			raw_data = read.get_raw_data(scale=True)
			if len(raw_data) >= (cutoff + cutlen):
					PNlist.append(read.read_id)
	return PNlist

@click.command()
@click.option('--target', '-t', help='The input fast5 folder path', type=click.Path(exists=True))
@click.option('--filename', '-f', help='The output file name', type=click.Path())
@click.option('--outfile', '-o', help='The output result folder path', type=click.Path())
@click.option('--cutoff', '-c', default=1500, help='Cutoff the first c signals')
@click.option('--cutlen', '-cut', default=10000, help='Cutoff the first c signals')

def main(target,filename,outfile,cutoff,cutlen):
	### make output folder
	if not os.path.exists(outfile):
		os.makedirs(outfile)

	### load data
	target_list = []

	print("making ID lists")
	for fileNM in glob.glob(target + '/*.fast5'):
		#print(fileNM)
		target_list = get_raw_data(target, fileNM, target_list, cutoff,cutlen)

	num_tar = 0
	with open(outfile + '/' + filename + '.txt', 'a') as f:
		for cont in target_list:
			num_tar +=1
			#print(cont)
			f.write(str(cont)+'\n')

	print("target : " + str(num_tar))
if __name__ == "__main__":
	main()