#!/bin/zsh

#batch size
id_dir="/z/kiku/Dataset/ID_three"
src_dir="/z/kiku/Dataset/Target_three"

class=3
model=$1
# #DIRECTORY CLEAN
# rm -rf $tar_dir/${IDlist}/*
# rm -rf $tar_dir/pytorch/*

#MAKE ID LIST
#python IDlist.py -t ${src_dir}/${positive} -o ${id_dir} -f $positive -cut ${cutlen}

#TRAIN
for i in 2000 3000 4000 5000
do
    python main.py -t $id_dir -i $src_dir -class $class -len $i -a $model
done
