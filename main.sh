#!/bin/zsh

#batch size
id_dir="/z/kiku/Dataset/ID"
src_dir="/z/kiku/Dataset/Target"

class=6
# #DIRECTORY CLEAN
# rm -rf $tar_dir/${IDlist}/*
# rm -rf $tar_dir/pytorch/*

#MAKE ID LIST
#python IDlist.py -t ${src_dir}/${positive} -o ${id_dir} -f $positive -cut ${cutlen}

##TRAIN
#for i in 3000 4000 5000;do
    #for j in ResNet LSTM;do
        #python main.py -t $id_dir -i $src_dir -class $class -len $i -b 700 -a $j -me 40
    #done
#done

python main.py -t $id_dir -i $src_dir -class 6 -len 3000 -b 1200 -a LSTM -me 20 -hidden 64
#for j in 64 128 256 512;do
    #python main.py -t $id_dir -i $src_dir -class $class -len 3000 -a LSTM -b 350 -me 50 -hidden $j
#done
