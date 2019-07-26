#!/bin/bash
python3.5 translate.py -gpu 0 \
                    -batch_size 20 \
                    -beam_size 10 \
                    -model models/wikihow_step_40000.pt \
                    -src tmp.txt \
                    -output testout/tmp.out \
                    -min_length 8 \
                    -verbose \
                    -stepwise_penalty \
                    -coverage_penalty summary \
                    -beta 5 \
                    -length_penalty wu \
                    -alpha 0.9 \
                    -verbose \
                    -block_ngram_repeat 3 \
                    -ignore_when_blocking "." "</t>" "<t>"
