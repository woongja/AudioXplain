#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python lime_deletion.py \
  --dataset_path /home/woongjae/XAI/Datasets/PartialSpoof/database/eval/con_wav \
  --protocol_path /home/woongjae/XAI/protocol/aasist_ps_spoof.txt \
  --batch_size 64 \
  --sr 16000 \
  --frame_ms 500 \
  --num_samples 200 \
  --log_file /home/woongjae/XAI/lime_delection/deletion.log \
  --output_csv /home/woongjae/XAI/result/results_deletion.csv
