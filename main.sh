#!/bin/zsh

#batch size
id_dir="/z/kiku/Dataset/ID"
src_dir="/z/kiku/Dataset/Target"

# #DIRECTORY CLEAN
# rm -rf $tar_dir/${IDlist}/*
# rm -rf $tar_dir/pytorch/*

#MAKE ID LIST
#python IDlist.py -t ${src_dir}/${positive} -o ${id_dir} -f $positive -cut ${cutlen}

##TRAIN
#for i in 3000 4000 5000;do
    #python main.py -t $id_dir -i $src_dir -class 6 -len $i -a Transformer -b 200 -me 40
#done

#python main.py -t $id_dir -i $src_dir -class 2 -len 5000 -b 200 -a GRU -me 40 -hidden 64 -t_class 1
python main.py -t $id_dir -i $src_dir -class 4 -len 7000 -b 100 -a ResNet -me 20 -t_class 0 
#python main.py -t $id_dir -i $src_dir -class 2 -len 5000 -b 200 -a Transformer -me 40
#for j in 50 70 90 ;do
    #for i in 64 128 256;do
        #python main.py -t $id_dir -i $src_dir -class 2 -len 5000 -a GRU -b 200 -me $j -hidden $i 
    #done
#done
