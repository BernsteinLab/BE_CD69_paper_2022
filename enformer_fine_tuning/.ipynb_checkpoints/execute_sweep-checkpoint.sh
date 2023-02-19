#!/bin/bash -l

python3 enformer_ft_train.py \
            --tpu_name="pod1" \
            --tpu_zone="us-east1-d" \
            --wandb_project="enformer_fine_tuning" \
            --wandb_user="njaved" \
            --wandb_sweep_name="enformer_fine_tuning" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/be_paper_finetuning/tfrecords" \
            --num_epochs=120 \
            --warmup_frac=0.025 \
            --patience=30\
            --min_delta=0.001 \
            --num_heads=3 \
            --model_save_dir="gs://picard-testing-176520/be_paper_finetuning/models" \
            --model_save_basename="enformer_fine_tuning_230118" \
            --lr_base1="5.0e-05" \
            --lr_base2="5.0e-03" \
            --weight_decay_frac1="5.0e-07" \
            --weight_decay_frac2="5.0e-07" \
            --dropout_rate="0.40" \
            --attention_dropout_rate="0.05" \
            --positional_dropout_rate="0.01" \
            --epsilon=1.0e-10 \
            --num_parallel=4 \
            --savefreq=4
