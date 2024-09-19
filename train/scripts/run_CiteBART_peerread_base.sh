#!/bin/bash

python ../train_base_cit_pred_BART.py --model_name "peerread_base_BART_epoch_15" --dataset_path "../../cit_data/peerread_base/" --auto_find_batch_size True --num_epochs 15 --max_token_limit 400 --pretrained_model_path "facebook/bart-base" --warmup_steps 500 --checkpoints_path "../../checkpoints" --models_path "../../models"



