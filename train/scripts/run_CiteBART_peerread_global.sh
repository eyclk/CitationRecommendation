#!/bin/bash

python ../train_global_cit_pred_BART.py --model_name "peerread_global_BART_epoch_30" --dataset_path "../../cit_data/peerread_global/" --auto_find_batch_size True --num_epochs 30 --max_token_limit 350 --pretrained_model_path "facebook/bart-base" --warmup_steps 500 --checkpoints_path "../../checkpoints" --models_path "../../models"



