# Valid, Plausible, and Diverse Retrosynthesis Using Tied Two-way Transformers with Latent Variables

This repository is an implementation of "Valid, Plausible, and Diverse Retrosynthesis Using Tied Two-way Transformers with Latent Variables" submitted to the Journal of Chemical Information and Modeling. The code is built on top of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).


## Requirements

- Python 3.6
- torch 1.4
- torchtext 0.4.0
- configargparse
- tqdm


## Preprocess

The USPTO-50k dataset is located in `./data/`. To train the model, the dataset needs to be preprocessed.

```
DATA=data

TRAIN_SRC=$DATA/src-train.txt
VALID_SRC=$DATA/src-val.txt
TRAIN_TGT=$DATA/tgt-train.txt
VALID_TGT=$DATA/tgt-val.txt

DATA_PREFIX=data/USPTO-50k_no_rxn_processed
python preprocess.py \
    -train_src $TRAIN_SRC \
    -train_tgt $TRAIN_TGT \
    -valid_src $VALID_SRC \
    -valid_tgt $VALID_TGT \
    -save_data $DATA_PREFIX \
    -share_vocab -overwrite
```


## Train

Train the model using the preprocessed dataset. In the paper, we tested the dimension of multinomial latent variable for 1, 2, and 5. In the code below, the dimension, `num_experts`, is set to 1. 

```
OUT="onmt-runs"
[ -d $OUT ] || mkdir -p $OUT
[ -d $OUT/model ] || mkdir -p $OUT/model

python train.py -data $DATA_PREFIX \
    -save_model $OUT/model/model -train_steps 500000 \
    -save_checkpoint_steps 5000 -keep_checkpoint 11 \
    -valid_step 5000 -report_every 5000 \
    -batch_size 4096 -batch_type tokens -normalization tokens \
    -max_relative_positions 4 -share_relative_pos_embeddings \
    -layers 6 -rnn_size 256 -word_vec_size 256 -heads 8 -transformer_ff 2048 \
    -num_experts 1 -position_encoding -share_embeddings \
    -dropout 0.3 -max_generator_batches 0 -early_stopping 40 \
    -gpu_ranks 0 -world_size 1 -accum_count 4 \
    -max_grad_norm 0 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 \
    -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -param_init 0 -param_init_glorot -seed 2020 \
    2>&1 | tee -a $OUT/model/train.log
```


After finishing the training, we averaged the last 5 models. For example, below is the code for averaging 5 models.

```
models="$OUT/model/model_step_20000.pt \
        $OUT/model/model_step_25000.pt \
        $OUT/model/model_step_30000.pt \
        $OUT/model/model_step_35000.pt \
        $OUT/model/model_step_40000.pt" 
MODEL="$OUT/model/model_step_40000_avg5.pt"

python average_models.py -models $models -output $MODEL
```


## Inference & Evaluation

You can use the code below to evaluate the model for a given test dataset.

```
DATA=data
TEST_SRC=$DATA/src-test.txt
TEST_TGT=$DATA/tgt-test.txt

MODEL="$OUT/model/model_step_*.pt"

TRANSLATE_OUT=$OUT/model/test/step_*
[ -d $TRANSLATE_OUT ] || mkdir -p $TRANSLATE_OUT

python translate.py -model $MODEL \
    -src $TEST_SRC -tgt $TEST_TGT \
    -output $TRANSLATE_OUT \
    -beam_size 20 -n_best 10 \
    -max_length 200 \
    -num_experts 1 \
    -batch_size 128 \
    -replace_unk -gpu 0
```


### Reproducing the results & Choosing the number of latent dimension

For those who want to reproduce the results in the paper, we have released one of the trained models which can be downloaded from <https://www.dropbox.com/s/57wdw8a937ruvwn/model_L2.pt?dl=0>. This model is trained with the latent dimension of `2`. After downloading the model in the `./onmt-runs/` directory, try the code below. Note that the `num_experts` option is set to `2` since the latent dimension of the model is `2`. As the dimension increases, the diversity in results improved (please refer the paper for details). However, there is a trade-off between the dimension and speed. So, we have tested the dimension for `1`, `2`, and `5`.

```
DATA=data
TEST_SRC=$DATA/src-test.txt
TEST_TGT=$DATA/tgt-test.txt

OUT="onmt-runs"
MODEL="$OUT/USPTO_L2.pt"

TRANSLATE_==$OUT/USPTO_L2_results
[ -d $TRANSLATE_OUT ] || mkdir -p $TRANSLATE_OUT

python translate.py -model $MODEL \
    -src $TEST_SRC -tgt $TEST_TGT \
    -output $TRANSLATE_OUT \
    -beam_size 10 -n_best 10 \
    -max_length 200 \
    -num_experts 2 \
    -batch_size 128 \
    -replace_unk -gpu 0
```

The translation and evaluation results can be found in `onmt-runs/USPTO_L2_results/`. Open `onmt-runs/USPTO_L2_results/pred_cycle_lp2.txt.score` to check the evaluation results. The top 10 rows show k (first column), top-k accuarcy (second column), and invalid SMILES rates (third column) while the last row is unique rate.

```
1,46.7226,0.059952
2,61.0312,0.069944
3,67.9856,0.113243
4,71.7426,0.194844
5,74.0008,0.335731
6,75.4396,0.516254
7,76.6787,0.845038
8,77.2982,1.46133
9,77.8377,2.62457
10,78.1775,4.60631
89.8321
```
