!#/bin/bash
python3 preprocess_copy.py -src /data/src.train.txt \
                           -tgt /data/tgt.train.txt \
                           -output data/train \
                           -prune 150 \
                           -num_examples 100000



python3 preprocess_copy.py -src /data/src.val.txt \
                           -tgt /data/tgt.val.txt \
                           -output data/val \
                           -prune 150 \
                           -num_examples 100000
