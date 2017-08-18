#!/bin/zsh

#src=out/gan-regen.loss
src=out/gan.loss

grep loss: $src|tail -100|python3 post_gan_loss.py \
  --stage_size 10  \
  --window_size 100 \
  --emp 1 


