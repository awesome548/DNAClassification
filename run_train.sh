#!/bin/zsh

positive=$1
negative=$2

#batch size
b1=$3
b2=$4
b3=$5
b4=$6
b5=$7

epoch=$8

layer="5"
depth="twice"

IDlist="ID"

src_dir="/z/kiku/Dataset"
tar_dir="/z/kiku/Basecaller/squiggle/RNN"
src_pos=$src_dir/$positive
src_neg=$src_dir/$negative
li_pos="${IDlist}/${positive}.txt"
li_neg="${IDlist}/${negative}.txt"

# #DIRECTORY CLEAN
rm -rf $tar_dir/${IDlist}/*
# rm -rf $tar_dir/pytorch/*

#MAKE ID LIST
python IDlist.py -t ${src_dir}/${positive} -o ${IDlist} -f $positive
for f in `ls -d ${src_dir}/${negative}*`;
do
    python IDlist.py -t $f -o $IDlist -f $negative
done    

#PREPROCESS
python Preprocess.py -gp $li_pos -gn $li_neg -i ${src_pos}train -o pytorch -b $b1
python Preprocess.py -gp $li_pos -gn $li_neg -i ${src_pos}validation -o pytorch -b $b2
python Preprocess.py -gp $li_pos -gn $li_neg -i ${src_neg}train -o pytorch -b $b3
python Preprocess.py -gp $li_pos -gn $li_neg -i ${src_neg}validation -o pytorch -b $b4

# #TRAIN
# echo python trainer.py \
# -pt pytorch/${positive}train_${b1}.pt -pv pytorch/${positive}validation_${b2}.pt \
# -nt pytorch/${negative}train_${b3}.pt -nv pytorch/${negative}validation_${b4}.pt \
# -o models/train_${positive}_${negative}_${layer}${depth}_${epoch}.ckpt -b $b5 -e $epoch

echo "positive : $positive negative : $negative , batch size : $b1  ,$b2 (positive) $b3,$b4 (negative)"

# #TEST
# echo python inference.py \
# -m models/train_${positive}_${negative}_${layer}${depth}_${epoch}.ckpt \
# -p ${src_pos}test -n ${src_neg}test \
# -b $b5