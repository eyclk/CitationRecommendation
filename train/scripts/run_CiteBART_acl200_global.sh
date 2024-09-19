#!/bin/bash

python ../train_global_cit_pred_BART.py --model_name "acl200_global_BART_epoch_15" --dataset_path "../../cit_data/acl200_global/" --auto_find_batch_size True --num_epochs 15 --max_token_limit 350 --pretrained_model_path "facebook/bart-base" --warmup_steps 500 --checkpoints_path "../../checkpoints" --models_path "../../models"



