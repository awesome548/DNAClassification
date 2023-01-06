#!/bin/zsh
wk_dir="/z/kiku/Basecaller/squiggle/RNN"
tar_dir="/z/kiku/Dataset/Target"
id_dir="/z/kiku/Dataset/ID"
cd $wk_dir
for i in A B C D E F;do
    python preprocess/IDlist.py -t $tar_dir/$i -o $id_dir -f $i
done