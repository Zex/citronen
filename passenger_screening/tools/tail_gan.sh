#!/bin/zsh

#src=gan-linear.loss
src=gan.loss

grep loss: $src|python3 post_gan_loss.py \
  --stage_size 100  \
  --window_size 100 \
  --emp 1 


