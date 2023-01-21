#!/bin/zsh
wk_dir="/z/kiku/Basecaller/RNN"
tar_dir="/z/kiku/Dataset/Target"
id_dir="/z/kiku/Dataset/ID"
cd $wk_dir

python IDlist.py -id $id_dir -in $tar_dir -c G
#for i in A B C D E F;do
    #python process/IDlist.py -t $tar_dir/$i -o $id_dir -f $i
#done