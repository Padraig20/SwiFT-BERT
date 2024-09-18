#!/bin/bash

# export NEPTUNE_API_TOKEN={YOUR_API_TOKEN}

cd /home/patrickstyll/Bachelorstudiengang-Software_and_Information_Engineering/SNU_Connectome_Lab/SwiFT-BERT/SwiFT-BERT # move to where 'SwiFUN' is located

TRAINER_ARGS="--accelerator cpu --max_epochs 10 --precision 16 --num_nodes 1 --devices 1 --strategy DDP"
MAIN_ARGS='--loggername tensorboard --dataset_name HBN --image_path /home/patrickstyll/Bachelorstudiengang-Software_and_Information_Engineering/SNU_Connectome_Lab/SwiFT-BERT/2.1.movieTP_MNI_to_TRs_znorm'
DATA_ARGS='--batch_size 2 --eval_batch_size 16 --num_workers 8'
DEFAULT_ARGS='--project_name {project_name}'
OPTIONAL_ARGS='--cope 5 --c_multiplier 2 --clf_head_version v1 --downstream_task emotion --use_scheduler --gamma 0.5 --cycle 0.7 --loss_type mse --last_layer_full_MSA True '  
RESUME_ARGS="" 

python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
  --dataset_split_num 1 --seed 1 --learning_rate 5e-5 --depths 2 2 6 2 --embed_dim 36 \
  --sequence_length 20 --first_window_size 4 4 4 1 --window_size 4 4 4 1 --img_size 96 96 96 255