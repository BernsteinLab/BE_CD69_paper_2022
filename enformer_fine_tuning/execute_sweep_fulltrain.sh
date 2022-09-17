#!/bin/bash -l

python3 enformer_ft_train_full.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="enformer_fine_tuning" \
            --wandb_user="njaved" \
            --wandb_sweep_name="enformer_fine_tuning" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/BE_paper_pretraining/tfrecords" \
            --num_epochs=100 \
            --warmup_frac=0.10 \
            --patience=6\
            --min_delta=0.001 \
            --num_heads=4 \
            --model_save_dir="gs://picard-testing-176520/BE_paper_pretraining/models" \
            --model_save_basename="enformer_fine_tuning" \
            --lr_base1="5.0e-06,5.0e-04" \
            --lr_base2="5.0e-02,5.0e-03" \
            --epsilon=1.0e-10 \
            --num_parallel=8 \
            --savefreq=10 
