#!/bin/bash -l

python3 enformer_ft_train.py \
            --tpu_name="node-15" \
            --tpu_zone="us-central1-a" \
            --wandb_project="enformer_fine_tuning" \
            --wandb_user="njaved" \
            --wandb_sweep_name="aformer_initial_run" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/BE_paper_pretraining/tfrecords" \
            --num_epochs=100 \
            --warmup_frac=0.10 \
            --patience=6\
            --min_delta=0.001 \
            --num_heads=4 \
            --model_save_dir="gs://picard-testing-176520/BE_paper_pretraining/models" \
            --model_save_basename="aformer_initial_tests" \
            --lr_base="5.0e-04" \
            --min_lr="5.0e-6" \
            --epsilon=1.0e-10 \
            --rectify=True \
            --slow_step_frac=0.5 \
            --sync_period=6 \
            --num_parallel=8 \
            --savefreq=8 \
            --use_fft_prior="True" \
            --freq_limit_scale="0.086" \
            --fft_prior_scale="0.25" 
