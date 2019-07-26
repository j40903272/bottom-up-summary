#!/bin/bash

mkdir -p $1
checkpoint="$1/$1_ckpt"
echo "################ checkpoint path : "$checkpoint
rm -rf $checkpoint
python3.7 -m allennlp.run train \
                          allennlp_config/$1.json \
                          --serialization-dir $checkpoint || { exit 1; }

predictionfile="$1/$1_pred"
python3.7 -m allennlp.run predict \
                          $checkpoint \
                          data/val.pred.txt \
                          --output $predictionfile \
                          --cuda-device 0 \
                          --batch-size 50 > /dev/null || { exit 1; }

outputsummary="$1/$1_summary"
python3.7 prediction_to_text.py -data $predictionfile \
                                -output $outputsummary \
                                -tgt data/val.src.txt \
                                -prune 500 > $1/score.txt || { exit 1; }

files2rouge $outputsummary /data/tgt.val.txt > $1/rouge.txt || { exit 1; }

clear
cat $1/score.txt
cat $1/rouge.txt
