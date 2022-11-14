#!/bin/zsh

positive=$1
negative=$2

#batch size
targetb1=$3
targetb2=$4
nontarb1=$5
nontarb2=$6
trainb=$7

epoch=$8


IDlist="ID"

src_dir="/z/kiku/Dataset/Target"
tar_dir="/z/kiku/Basecaller/squiggle/RNN"
src_pos=$src_dir/$positive
src_neg=$src_dir/$negative
li_pos="${IDlist}/${positive}.txt"
li_neg="${IDlist}/${negative}.txt"

cutlen=$9

# #DIRECTORY CLEAN
# rm -rf $tar_dir/${IDlist}/*
# rm -rf $tar_dir/pytorch/*

#MAKE ID LIST
python IDlist.py -t ${src_dir}/${positive} -o ${IDlist} -f $positive
# for f in `ls -d ${src_dir}/${negative}*`;
# do
#    python IDlist.py -t $f -o $IDlist -f $negative
# done

#PREPROCESS
# python Preprocess.py -gp $li_pos -gn $li_neg -i ${src_pos}train -o pytorch -b $targetb1 -l $cutlen
# python Preprocess.py -gp $li_pos -gn $li_neg -i ${src_pos}validation -o pytorch -b $targetb2 -l $cutlen
# python Preprocess.py -gp $li_pos -gn $li_neg -i ${src_pos}test -o pytorch -b $targetb2 -l $cutlen
# python Preprocess.py -gp $li_pos -gn $li_neg -i ${src_neg}train -o pytorch -b $nontarb1 -l $cutlen
# python Preprocess.py -gp $li_pos -gn $li_neg -i ${src_neg}validation -o pytorch -b $nontarb2 -l $cutlen
# python Preprocess.py -gp $li_pos -gn $li_neg -i ${src_neg}test -o pytorch -b $nontarb2 -l $cutlen

#TRAIN
# python main.py \
# -pt pytorch/${positive}train_${targetb1}.pt -pv pytorch/${positive}validation_${targetb2}.pt \
# -nt pytorch/${negative}train_${nontarb1}.pt -nv pytorch/${negative}validation_${nontarb2}.pt \
# -ptt pytorch/${positive}test_${targetb2}.pt -ntt pytorch/${negative}test_${nontarb2}.pt \
# -b $trainb -e $epoch -c $cutlen

echo "positive : $positive negative : $negative , batch size : $targetb1  ,$targetb2 (positive) $nontarb1,$nontarb2 (negative)"
