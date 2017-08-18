#!/bin/zsh

model="gan.py"

python3 $model \
  --data_root ../data/passenger_screening/stage1_aps \
  --label_path ../data/passenger_screening/stage1_labels.csv \
  --model_root ../models \
  --model_id pgscr-gan-regen \
  --lr 0.001 \
  --outpath gan_output_regen
  
