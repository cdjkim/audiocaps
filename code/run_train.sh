#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export LD_PRELOAD="/usr/lib/libtcmalloc.so"
export BETTER_EXCEPTIONS=1

num_gpus=1
batch_size=128

feature_name="fc3c4_pyramid"

# aligned semantic attention param
##############################
attention_flow_type="full"
use_context_lstm=true
context_concat=true

word_encoder_dropout=0.25
word_encoder_num_units=128

context_encoder_dropout=0.2
context_encoder_num_units=128

##############################

# encoder
encoder_dropout=0.25
encoder_unit_type="layer_norm_lstm"
pass_hidden_state=false
L0_encoder_type="simple_brnn"
L0_encoder_num_units=128

method="pyramid_semal_attention"
combine="add"

L1_encoder_type="simple_brnn"
L1_encoder_num_units=128

L2_encoder_type="simple_brnn"
L2_encoder_num_units=128
L3_encoder_type="off"
L3_encoder_num_units=0

# decoder
decoder_dropout=0.15
decoder_type="attention_rnn"
decoder_dropout=0.25
decoder_type="attention_rnn"
decoder_unit_type="layer_norm_lstm"
decoder_num_units=128

beam_width=1

# lr
init_lr=0.001
num_epochs_per_decay=15
lr_decay_factor=1.0
max_grad_norm=0.4
label_smoothing=0.02

# Logging
evaluation_epoch=0.3

# etc
data_name="audiocaps"
if [ $data_name = "audiocaps" ]; then
    vocab_size=4506
fi
max_length=51

# Data/checkpoints
model_name="PyramidLSTM"
other_info="${method}_L0unts${L0_encoder_num_units}.${L1_encoder_num_units}._decunit${decoder_num_units}"
other_info+="_ls${label_smoothing}"
current_time="`date +%m%d%H%M%S`";
checkpoint_dir="./checkpoints/${data_name}/${model_name}/${current_time}_${other_info}"

python train.py \
    --num_gpus $num_gpus \
    --init_lr $init_lr \
    --batch_size $batch_size \
    --model_name $model_name \
    --data_name $data_name \
    --other_info $other_info \
    --checkpoint_dir $checkpoint_dir \
    --vocab_size $vocab_size \
    --num_epochs_per_decay $num_epochs_per_decay \
    --lr_decay_factor $lr_decay_factor \
    --evaluation_epoch $evaluation_epoch \
    --max_grad_norm $max_grad_norm \
    --max_length $max_length \
    --label_smoothing $label_smoothing \
    --encoder_dropout $encoder_dropout \
    --encoder_unit_type $encoder_unit_type \
    --decoder_dropout $decoder_dropout \
    --decoder_unit_type $decoder_unit_type \
    --decoder_type $decoder_type \
    --decoder_num_units $decoder_num_units \
    --beam_width $beam_width \
    --pass_hidden_state $pass_hidden_state \
    --method $method \
    --combine $combine \
    --L0_encoder_type $L0_encoder_type \
    --L0_encoder_num_units $L0_encoder_num_units \
    --L1_encoder_type $L1_encoder_type \
    --L1_encoder_num_units $L1_encoder_num_units \
    --L2_encoder_type $L2_encoder_type \
    --L2_encoder_num_units $L2_encoder_num_units \
    --L3_encoder_type $L3_encoder_type \
    --L3_encoder_num_units $L3_encoder_num_units \
    --feature_name $feature_name \
    --attention_flow_type $attention_flow_type \
    --use_context_lstm $use_context_lstm \
    --context_concat $context_concat \
    --word_encoder_dropout $word_encoder_dropout \
    --word_encoder_num_units $word_encoder_num_units \
    --context_encoder_dropout $context_encoder_dropout \
    --context_encoder_num_units $context_encoder_num_units

