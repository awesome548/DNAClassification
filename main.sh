#!/bin/zsh

#batch size
id_dir="/z/kiku/Dataset/ID_four"
src_dir="/z/kiku/Dataset/Target_four"

class=2
model=$1
# #DIRECTORY CLEAN
# rm -rf $tar_dir/${IDlist}/*
# rm -rf $tar_dir/pytorch/*

#MAKE ID LIST
#python IDlist.py -t ${src_dir}/${positive} -o ${id_dir} -f $positive -cut ${cutlen}

#python main.py -t $id_dir -i $src_dir -class 2 -len 3000 -a ResNet -e 40 -b 1000
##TRAIN
#python main.py -t $id_dir -i $src_dir -class $class -len 3000 -a ResNet -b 700 -me 40 -e 60
#for i in 3000 4000 5000;do
    #for j in ResNet LSTM ;do
        #python main.py -t $id_dir -i $src_dir -class $class -len $i -a $j -b 700 -me 30 -e 60
    #done
#done

for i in 3000 4000 5000;do
    for j in 20 30 40 50;do
        python main.py -t $id_dir -i $src_dir -class $class -len $i -a ResNet -b 700 -me $j -e 100
    done
done
for i in 3000 4000 5000;do
    for j in 20 30 40 50;do
        python main.py -t $id_dir -i $src_dir -class $class -len $i -a LSTM -b 700 -me $j -e 100
    done
done