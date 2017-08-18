#!/bin/zsh

model="gan_tf.py"

python3 $model \
  --data_root ../data/passenger_screening/stage1_aps \
  --label_path ../data/passenger_screening/stage1_labels.csv \
  --model_root ../models-gan-tf \
  --model_id pgscr-gan-tf \
  --lr 0.001 \
  --outpath gan_tf_output
  
