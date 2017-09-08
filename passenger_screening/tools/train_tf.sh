#!/bin/zsh

model="with_tf.py"

python3 $model \
  --batch_size 16 \
  --data_root ../data/passenger_screening/stage1_aps \
  --label_path ../data/passenger_screening/stage1_labels.csv \
  --model_root ../models-conv \
  --model_id pgscr-tf \
  --lr 0.001 \
  --epochs 1000 \
  --init_step 0 
  
