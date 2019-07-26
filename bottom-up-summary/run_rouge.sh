#!/bin/bash

arr=("multi_attention" "stacked_attention" "intra" "alternating_lstm" "qanet")
 
for ((i=0; i < ${#arr[@]}; i++))
do
    echo $i ${arr[$i]}
    files2rouge ${arr[$i]}/${arr[$i]}_summary /data/tgt.val.txt > ${arr[$i]}/rouge.txt
done
