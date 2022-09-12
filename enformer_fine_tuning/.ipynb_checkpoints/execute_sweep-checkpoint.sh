#!/bin/bash -l

python3 enformer_ft_train.py \
            --tpu_name="node-15" \
            --tpu_zone="us-central1-a" \
            --wandb_project="enformer_fine_tuning" \
            --wandb_user="njaved" \
            --wandb_sweep_name="aformer_initial_run" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/BE_paper_pretraining/tfrecords" \
            --num_epochs=30 \
            --warmup_frac=0.15 \
            --patience=6\
            --min_delta=0.001 \
            --num_heads=4 \
            --model_save_dir="gs://picard-testing-176520/BE_paper_pretraining/models" \
            --model_save_basename="aformer_initial_tests" \
            --lr_base="5.0e-04" \
            --min_lr="5.0e-8" \
            --gradient_clip="0.2" \
            --weight_decay_frac="1.0e-05" \
            --epsilon=1.0e-10 \
            --rectify=True \
            --slow_step_frac=0.5 \
            --sync_period=6 \
            --num_parallel=768 \
            --savefreq=4 \
            --use_fft_prior="True" \
            --freq_limit_scale="0.07" \
            --fft_prior_scale="0.20" 