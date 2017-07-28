#!/bin/zsh

model="with_tf.py"

python3 $model \
  --data_root ../data/passenger_screening/stage1_a3daps \
  --label_path ../data/passenger_screening/stage1_labels.csv \
  --model_root ../models \
  --model_id pgscr-d3-tf \
  --lr 0.001 \
  --epochs 1000 \
  --init_step 0 
  
