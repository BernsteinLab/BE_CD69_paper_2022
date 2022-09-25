#!/bin/bash -l

python3 enformer_ft_train_enformer_calico.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="enformer_calico" \
            --wandb_user="njaved" \
            --wandb_sweep_name="enformer_calico" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/enformer_calico/tfrecords" \
            --num_epochs=120 \
            --warmup_frac=0.10 \
            --patience=30\
            --min_delta=0.001 \
            --num_heads=7 \
            --model_save_dir="gs://picard-testing-176520/enformer_calico/models" \
            --model_save_basename="enformer_calico" \
            --lr_base1="5.0e-05" \
            --lr_base2="5.0e-03" \
            --weight_decay_frac1="5.0e-05" \
            --weight_decay_frac2="5.0e-02" \
            --dropout_rate="0.40" \
            --attention_dropout_rate="0.05" \
            --positional_dropout_rate="0.01" \
            --epsilon=1.0e-10 \
            --num_parallel=8 \
            --savefreq=8 
