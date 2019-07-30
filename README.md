# Bottom-Up-Summary
https://arxiv.org/pdf/1808.10792.pdf

https://github.com/sebastianGehrmann/bottom-up-summary

http://opennmt.net/OpenNMT-py/Summarization.html

https://github.com/harvardnlp/sent-summary

---

# Extractive
python3.7
https://github.com/sebastianGehrmann/bottom-up-summary

### preprocess
```
$1 = train, val
python preprocess_copy.py -src /data/src.$1.txt \
                          -tgt /data/tgt.$1.txt \
                          -output data/$1
                          -prune 500 \
                          -num_examples 100000
```
src, tgt --> line pair --> text, summary

### set config
* modify allennlp_config/*.json
"train_data_path" -> data/train.txt
"validation_data_path" -> data/val.txt
* elmo_tagger*.json
```
wget
http://nlp.stanford.edu/data/glove.6B.zip
unzip
"pretrained_file" -> "glove.6B.100d.txt.gz"
```

```
wget https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
wget https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
"options_file" -> "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
"weight_file" -> "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
```
### train tagging model
```
python3 -m allennlp.run train
           allennlp_config/simple_tagger.json \
           --serialization-dir $model_ckpt (model checkpoint dir name)
```
						
### inference
```
python3 -m allennlp.run predict \
                        $model_ckpt \
                        data/val.pred.txt \
                        --output $predictionfile \
                        --cuda-device 0 \
                        --batch-size 50
```
### use as extractive summarizer
```
python3 prediction_to_text.py -data $predictionfile\
                              -output $summary \
                              -tgt data/val.src.txt\
                              -prune 500 \
                              -divider "" \
                              -threshold 0.25
```
### handle crf output
* add softmax function
```
def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
```
* prediction_to_text.py line:157
```
probs = [softmax(p)[1] for p in cline["logits"][:len(words)]]
```


### data & format
 * /data  (line pairs)
src.train.txt
src.val.txt
tgt.train.txt
tgt.val.txt
* bottom-up-summary/data/  (preprocess data)
train.pred.txt : json {"sentence":""}
train.src.txt  : a line of 0/1
train.txt      : [word]###[tag]
val.pred.txt
val.src.txt
val.txt

### openai-gpt config example
```
{
  "dataset_reader": {
    "type": "sequence_tagging",
    "word_tag_delimiter": "###",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "elmo": {
        "type": "elmo_characters"
      },
      "openai_transformer": {
          "type": "openai_transformer_byte_pair",
          "model_path": "https://s3-us-west-2.amazonaws.com/allennlp/models/openai-transformer-lm-2018.07.23.tar.gz"
      }
    }
  },
  "train_data_path": "/home/bottom-up-summary/data/train.txt",
  "validation_data_path": "/home/bottom-up-summary/data/val.txt",
  "model": {
    "type": "crf_tagger",
    "text_field_embedder": {
        "allow_unmatched_keys":true,
        "embedder_to_indexer_map": {
            "tokens": ["tokens"],
            "elmo": ["elmo"],
            "openai_transformer": ["openai_transformer", "openai_transformer-offsets"]
        },
        "token_embedders": {
            "tokens": {
                    "type": "embedding",
                    "embedding_dim": 100,
                    "pretrained_file": "/home/bottom-up-summary/data/glove.6B.100d.txt",
                    "trainable": true
            },
            "elmo": {
                "type": "elmo_token_embedder",
                "options_file": "/home/bottom-up-summary/data/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                "weight_file": "/home/bottom-up-summary/data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                "do_layer_norm": true,
                "dropout": 0.5
            },
            "openai_transformer": {
                "type": "openai_transformer_embedder",
                "transformer": {
                    "model_path": "https://s3-us-west-2.amazonaws.com/allennlp/models/openai-transformer-lm-2018.07.23.tar.gz"
                }
            }
        }
    },
    "encoder": {
            "type": "lstm",
            "input_size": 1892, // 100 + 1024 + 768
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.5,
            "bidirectional": true
    }
  },
  "iterator": {"type": "basic", "batch_size": 32},
  "trainer": {
    "optimizer": "adagrad",
    "grad_clipping": 2.0,
    "num_epochs": 10,
    "patience": 3,
    "cuda_device": 0
  }
  "regularizer": [
    ["transitions$", {"type": "l2", "alpha": 0.01}]
  ]
```

### experiment results
* pretrain embedding

| embedding          | AUC  | F1    | ROUGE-1-R | ROUGE-1-P | ROUGE-1-F | ROUGE-2-R | ROUGE-2-P | ROUGE-2-F | ROUGE-L-R | ROUGE-L-P | ROUGE-L-F |
|--------------------|------|-------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| no-pretrain        | 88.7 | 49.64 | 0.44484   | 0.34150   | 0.36073   | 0.04229   | 0.04555   | 0.04076   | 0.33543   | 0.25414   | 0.26925   |
| glove elmo         | 89.2 | 54.9  | 0.45566   | 0.41775   | 0.40695   | 0.07282   | 0.09278   | 0.07624   | 0.34141   | 0.31411   | 0.30428   |
| glove elmo crf     | 90.9 | 33.19 | 0.09488   | 0.57578   | 0.16022   | 0.01772   | 0.13388   | 0.03081   | 0.07651   | 0.47249   | 0.12953   |
| glove elmo gpt     | 91.5 | 58.18 | 0.47133   | 0.42165   | 0.42038   | 0.07363   | 0.08603   | 0.07448   | 0.35354   | 0.31829   | 0.31551   |
| glove elmo gpt crf | 91.2 | 52.91 | 0.34443   | 0.42786   | 0.36693   | 0.03641   | 0.05487   | 0.04181   | 0.25310   | 0.31532   | 0.26967   |

* seq2seq architecture

| architecture              | AUC  | F1    | ROUGE-1-R | ROUGE-1-P | ROUGE-1-F | ROUGE-2-R | ROUGE-2-P | ROUGE-2-F | ROUGE-L-R | ROUGE-L-P | ROUGE-L-F |
|---------------------------|------|-------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| lstm                      | 88.7 | 49.64 | 0.44484   | 0.34150   | 0.36073   | 0.04229   | 0.04555   | 0.04076   | 0.33543   | 0.25414   | 0.26925   |
| alternating_lstm          | 88.9 | 48.23 | 0.50814   | 0.31396   | 0.36480   | 0.03967   | 0.03348   | 0.03396   | 0.39625   | 0.24167   | 0.28148   |
| intra sentence attention  | 84.4 | 33.33 | 0.23919   | 0.20699   | 0.20166   | 0.00998   | 0.01143   | 0.00982   | 0.22858   | 0.19844   | 0.19275   |
| multi head self attention | 84.1 | 27.2  | 0.18059   | 0.18246   | 0.16640   | 0.00956   | 0.01441   | 0.01069   | 0.17076   | 0.17368   | 0.15777   |
| stacked attention         | 88.9 | 49.28 | 0.52421   | 0.29309   | 0.35853   | 0.02421   | 0.01542   | 0.01784   | 0.40115   | 0.21991   | 0.27053   |
| transformer               | 88.2 | 46.97 | 0.52671   | 0.29341   | 0.35236   | 0.02818   | 0.02016   | 0.02207   | 0.38912   | 0.22317   | 0.26923   |
| qanet                     | 89.0 | 52.94 | 0.54186   | 0.29535   | 0.36389   | 0.03603   | 0.02552   | 0.02807   | 0.42458   | 0.22859   | 0.28240   |

* headline

| elmo tagger               | AUC  | F1    | ROUGE-1-R | ROUGE-1-P | ROUGE-1-F | ROUGE-2-R | ROUGE-2-P | ROUGE-2-F | ROUGE-L-R | ROUGE-L-P | ROUGE-L-F |
|---------------------------|------|-------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| lstm                      | 83.3 | 51.34 | 0.34924   | 0.46079   | 0.37230   | 0.08633   | 0.13026   | 0.09742   | 0.31354   | 0.42041   | 0.33665   |


### encoder architecture
* lstm, gru, rnn
* alternating lstm
* augmented_lstm
* stacked_birectional_lstm
* bidirectional_language_model_transformer
* gated-cnn-encoder
* intra_sentence_attention
* multi_head_self_attention
* stacked_self_attention
* qanet_encoder
* pass_through
* feedforward

---
# Abstractive
python3.5
http://opennmt.net/OpenNMT-py/Summarization.html

### preprocess
dataset : https://github.com/harvardnlp/sent-summary

(1) wikihow
```
python preprocess.py -train_src /data/src.train.txt \
                     -train_tgt /data/tgt.train.txt \
                     -valid_src /data/src.val.txt \
                     -valid_tgt /data/tgt.val.txt \
                     -save_data data/wikihow \
                     -src_seq_length 10000 \
                     -tgt_seq_length 10000 \
                     -src_seq_length_trunc 500 \
                     -tgt_seq_length_trunc 100 \
                     -dynamic_dict \
                     -share_vocab \
                     -shard_size 100000
```
(2) cnndm
```
python preprocess.py -train_src data/cnndm/train.txt.src \
                     -train_tgt data/cnndm/train.txt.tgt.tagged \
                     -valid_src data/cnndm/val.txt.src \
                     -valid_tgt data/cnndm/val.txt.tgt.tagged \
                     -save_data data/cnndm/CNNDM \
                     -src_seq_length 10000 \
                     -tgt_seq_length 10000 \
                     -src_seq_length_trunc 400 \
                     -tgt_seq_length_trunc 100 \
                     -dynamic_dict \
                     -share_vocab \
                     -shard_size 100000
```
(3) Gigaword
```
python preprocess.py -train_src data/giga/train.article.txt \
                     -train_tgt data/giga/train.title.txt \
                     -valid_src data/giga/valid.article.txt \
                     -valid_tgt data/giga/valid.title.txt \
                     -save_data data/giga/GIGA \
                     -src_seq_length 10000 \
                     -dynamic_dict \
                     -share_vocab \
                     -shard_size 100000
```
### train
(1) wikihow
```
python train.py -save_model models/wikihow \
                -data data/wikihow \
                -copy_attn \
                -global_attention mlp \
                -word_vec_size 128 \
                -rnn_size 512 \
                -layers 1 \
                -encoder_type brnn \
                -train_steps 2000000 \
                -max_grad_norm 2 \
                -dropout 0. \
                -batch_size 64 \
                -valid_batch_size 64 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -reuse_copy_attn \
                -copy_loss_by_seqlength \
                -bridge \
                -seed 7122 \
                -world_size 1 \
                -gpu_ranks 0 \
                -early_stopping 5 \
                -report_every 200 \
                -log_file dada_exp/wikihow_log \
                -tensorboard \
                -tensorboard_log_dir dada_exp/wikihow_tensorboard_log
```
(2) cnndm
```
python train.py -save_model models/cnndm \
                -data data/cnndm/CNNDM \
                -copy_attn \
                -global_attention mlp \
                -word_vec_size 128 \
                -rnn_size 512 \
                -layers 1 \
                -encoder_type brnn \
                -train_steps 200000 \
                -max_grad_norm 2 \
                -dropout 0. \
                -batch_size 16 \
                -valid_batch_size 16 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -reuse_copy_attn \
                -copy_loss_by_seqlength \
                -bridge \
                -seed 777 \
                -world_size 1 \
                -gpu_ranks 0 \
                -early_stopping 5 \
                -report_every 200 \
                -log_file dada_exp/cnndm_log \
                -tensorboard \
                -tensorboard_log_dir dada_exp/cnndm_tensorboard_log
```
(3) Transformer
```
python -u train.py -data data/cnndm/CNNDM \
                   -save_model models/cnndm \
                   -layers 4 \
                   -rnn_size 512 \
                   -word_vec_size 512 \
                   -max_grad_norm 0 \
                   -optim adam \
                   -encoder_type transformer \
                   -decoder_type transformer \
                   -position_encoding \
                   -dropout 0\.2 \
                   -param_init 0 \
                   -warmup_steps 8000 \
                   -learning_rate 2 \
                   -decay_method noam \
                   -label_smoothing 0.1 \
                   -adam_beta2 0.998 \
                   -batch_size 4096 \
                   -batch_type tokens \
                   -normalization tokens \
                   -max_generator_batches 2 \
                   -train_steps 200000 \
                   -accum_count 4 \
                   -share_embeddings \
                   -copy_attn \
                   -param_init_glorot \
                   -world_size 1 \
                   -gpu_ranks 0 \
                   -early_stopping 5 \
                   -report_every 200 \
                   -log_file dada_exp/cnndm_transformer_log \
                   -tensorboard \
                   -tensorboard_log_dir dada_exp/cndmm_tensorboard_log
```

(4) Gigaword
```
python train.py -data data/giga/GIGA \
                -save_model models/giga \
                -copy_attn \
                -reuse_copy_attn \
                -train_steps 200000
```

### inference
(1) wikihow
```
python translate.py -gpu 0 \
                    -batch_size 20 \
                    -beam_size 10 \
                    -model models/wikihow_step_40000.pt \  ### select one model
                    -src /data/src.val.txt \
                    -output testout/wikihow.out \
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
```
(2) cnndm
```
python translate.py -gpu 0 \
                    -batch_size 20 \
                    -beam_size 10 \
                    -model models/cnndm... \
                    -src data/cnndm/test.txt.src \
                    -output testout/cnndm.out \
                    -min_length 35 \
                    -verbose \
                    -stepwise_penalty \
                    -coverage_penalty summary \
                    -beta 5 \
                    -length_penalty wu \
                    -alpha 0.9 \
                    -block_ngram_repeat 3 \
                    -ignore_when_blocking "." "</t>" "<t>"
```


### evaluation
(1) wikihow
files2rouge : https://github.com/pltrdy/files2rouge
```
files2rouge wikihow.out /data/tgt.val.txt
```

(2) cnndm
pyrouge : https://github.com/falcondai/pyrouge
rouge-basline : https://github.com/sebastianGehrmann/rouge-baselines
```
python baseline.py -s ../OpenNMT-py/testout/cnndm.out \
                   -t ../OpenNMT-py/data/cnndm/test.txt.tgt.tagged \
                   -m sent_tag_verbatim -r
```
(3) Gigaword
```
files2rouge giga.out test.title.txt --verbose
```


### experiment results
(1) wikihow
* copy attention
* encoder : brnn
* decoder : rnn
* emb : 128
* hidden : 512

|         |           |         |                                 |
|---------|-----------|---------|---------------------------------|
| ROUGE-1 | Average-R | 0.43287 | 95%-conf.int. 0.43065 - 0.43521 |
| ROUGE-1 | Average-P | 0.56836 | 95%-conf.int. 0.56518 - 0.57160 |
| ROUGE-1 | Average-F | 0.48303 | 95%-conf.int. 0.48068 - 0.48546 |
| ROUGE-2 | Average-R | 0.22634 | 95%-conf.int. 0.22423 - 0.22847 |
| ROUGE-2 | Average-P | 0.31730 | 95%-conf.int. 0.31408 - 0.32028 |
| ROUGE-2 | Average-F | 0.25812 | 95%-conf.int. 0.25566 - 0.26039 |
| ROUGE-L | Average-R | 0.42426 | 95%-conf.int. 0.42215 - 0.42659 |
| ROUGE-L | Average-P | 0.55834 | 95%-conf.int. 0.55520 - 0.56150 |
| ROUGE-L | Average-F | 0.47392 | 95%-conf.int. 0.47165 - 0.47628 |

(2) cnndm
* copy attention
* encoder : brnn
* decoder : rnn
* emb : 128
* hidden : 512

|         |           |         |                                 |
|---------|-----------|---------|---------------------------------|
| ROUGE-1 | Average-R | 0.40427 | 95%-conf.int. 0.40173 - 0.40666 |
| ROUGE-1 | Average-P | 0.39331 | 95%-conf.int. 0.39053 - 0.39616 |
| ROUGE-1 | Average-F | 0.38322 | 95%-conf.int. 0.38107 - 0.38527 |
| ROUGE-2 | Average-R | 0.16980 | 95%-conf.int. 0.16751 - 0.17192 |
| ROUGE-2 | Average-P | 0.16949 | 95%-conf.int. 0.16707 - 0.17192 |
| ROUGE-2 | Average-F | 0.16280 | 95%-conf.int. 0.16060 - 0.16480 |
| ROUGE-L | Average-R | 0.37091 | 95%-conf.int. 0.36847 - 0.37316 |
| ROUGE-L | Average-P | 0.36163 | 95%-conf.int. 0.35892 - 0.36436 |
| ROUGE-L | Average-F | 0.35199 | 95%-conf.int. 0.34979 - 0.35398 |


