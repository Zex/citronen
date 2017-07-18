#!/bin/zsh

model="with_torch.py"
if [ ! -z $1 ] ;then model=$1;fi

python3 $model \
  --data_root ../data/passenger_screening/stage1_a3daps \
  --label_path ../data/passenger_screening/stage1_labels.csv \
  --model_root ../models \
  --model_id pgscr-d3
 # --chkpt ../models/pgscr.chkpt-19952-0.1062
  
