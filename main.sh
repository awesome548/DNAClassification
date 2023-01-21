#!/bin/zsh

#batch size
#id_dir="/z/kiku/Dataset/ID"
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

#python test.py -id $id_dir -i $src_dir
python main.py -id $id_dir -i $src_dir -class 4 -len 3000 -b 500 -a ResNet -me 25 -t_class 1 -lr 0.01
#python main.py -id $id_dir -i $src_dir -class 6 -len 3000 -b 500 -a ResNet -me 30 -t_class 5
#python main.py -id $id_dir -i $src_dir -class 6 -len 3000 -b 500 -a GRU -me 30 -hidden 128 -t_class 1
#python main.py -id $id_dir -i $src_dir -class 6 -len 3000 -b 200 -a Transformer -me 30
#python main.py -id $id_dir -i $src_dir -class 2 -len 9000 -b 100 -a Effnet -me 30 -t_class 0 -lr 0.002463
#for j in 50 70 90 ;do
    #for i in 64 128 256;do
        #python main.py -t $id_dir -i $src_dir -class 2 -len 5000 -a GRU -b 200 -me $j -hidden $i
    #done
#done
