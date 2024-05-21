# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2021 THUML @ Tsinghua University
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the TimesNet (https://arxiv.org/abs/2210.02186) implementation
# from https://github.com/thuml/Time-Series-Library by THUML @ Tsinghua University
####################################################################################

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL/ \
  --model_id MSL \
  --model TimesNet \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 8 \
  --d_ff 16 \
  --e_layers 1 \
  --enc_in 55 \
  --c_out 55 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 1 \
  --seed 1