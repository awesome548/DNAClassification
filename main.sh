#!/bin/zsh

positive=$1

#batch size

id_dir="/z/kiku/Dataset/ID_four"
src_dir="/z/kiku/Dataset/Target_four"

cutlen=$2
class=$3

# #DIRECTORY CLEAN
# rm -rf $tar_dir/${IDlist}/*
# rm -rf $tar_dir/pytorch/*

#MAKE ID LIST
#python IDlist.py -t ${src_dir}/${positive} -o ${id_dir} -f $positive -cut ${cutlen}

#TRAIN
python main.py -t $id_dir -i $src_dir -class $class -cut $cutlen

