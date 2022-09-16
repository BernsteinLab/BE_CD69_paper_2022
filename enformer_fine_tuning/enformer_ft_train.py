import time
import os
import subprocess
import sys
import re
import argparse
import collections
import gzip
import math
import shutil
import matplotlib.pyplot as plt
import wandb
import numpy as np
import time
import pandas as pd
from datetime import datetime
import random

#import logging
#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf
import sonnet as snt
import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision

## custom modules
import enformer as enformer
import metrics as metrics
import training_utils

import seaborn as sns
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr  
from scipy import stats

import optimizers

 ## reformat 
# ===========================================================================#

def main():
    # ============== arg parse ==============================================# 
    parser = argparse.ArgumentParser(
        description='process input for genformer training loop')
    parser = training_utils.parse_args(parser)
    args = parser.parse_args()
    
    #================ init ==================================================# 
    
    ### make sure gcloud auth set to picard-testing-176520
        
    ### make sure TPU started

    # ============== define sweep options ==================== #
    sweep_config = {
            "name" : args.wandb_sweep_name,
            'method': "grid",
            'metric': {
                'name': 'hg_val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'lr_base': {
                    'values':[float(x) for x in args.lr_base.split(',')]
                },
                'min_lr': {
                    'values':[float(x) for x in args.min_lr.split(',')]
                },
                'gradient_clip': {
                    'values': [float(x) for x in args.gradient_clip.split(',')]
                },
                'epsilon': {
                    'values':[args.epsilon]
                },
                'rectify': {
                    'values':[args.rectify]
                },
                'use_fft_prior': {
                    'values':[x == 'True' for x in args.use_fft_prior.split(',')]
                },
                'fft_prior_scale': {
                    'values':[float(x) for x in args.fft_prior_scale.split(',')]
                },
                'freq_limit_scale': {
                    'values':[float(x) for x in args.freq_limit_scale.split(',')]
                },
                'beta1': {
                    'values':[float(x) for x in args.beta1.split(',')]
                },
                'beta2': {
                    'values':[float(x) for x in args.beta2.split(',')]
                }
                }

    }

    
    def sweep_train(config_defaults=None):
        # Set default values
        # Specify the other hyperparameters to the configuration, if any

        ## tpu initialization
        strategy = training_utils.tf_tpu_initialize(args.tpu_name)

        ## rest must be w/in strategy scope
        with strategy.scope():
            config_defaults = {
                "lr_base": 0.01 ### will be overwritten
            }
            
            ### log training parameters
            wandb.init(config=config_defaults, 
                       project= args.wandb_project, 
                       entity=args.wandb_user)
            #wandb.init(mode="disabled")
            wandb.config.tpu=args.tpu_name
            wandb.config.gcs_path=args.gcs_path
            wandb.config.input_length=args.input_length
            wandb.config.num_epochs=args.num_epochs
            #wandb.config.train_steps=args.train_steps
            #wandb.config.val_steps=args.val_steps
            wandb.config.warmup_frac=args.warmup_frac
            #wandb.config.total_steps=args.total_steps
            wandb.config.patience=args.patience
            wandb.config.min_delta=args.min_delta
            wandb.config.model_save_dir=args.model_save_dir
            wandb.config.model_save_basename=args.model_save_basename
            
            wandb.config.sync_period=args.sync_period
            wandb.config.slow_step_frac=args.slow_step_frac
            
            wandb.run.name = '_'.join(['LR' + str(wandb.config.lr_base),
                                        'FFTscale-' + str(wandb.config.fft_prior_scale),
                                        'FFTfreq-' + str(wandb.config.freq_limit_scale)])
            '''
            TPU init options
            '''
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy=\
                tf.data.experimental.AutoShardPolicy.FILE
            options.deterministic=False
            options.experimental_threading.max_intra_op_parallelism=1
            tf.config.optimizer.set_jit(True)

            NUM_REPLICAS = strategy.num_replicas_in_sync
            BATCH_SIZE_PER_REPLICA=1
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA*NUM_REPLICAS
            
            train_steps = 5#34021 // GLOBAL_BATCH_SIZE
            val_steps = 3#2213 // GLOBAL_BATCH_SIZE
            
            total_steps = train_steps * wandb.config.num_epochs
            
            tr_data,val_data = training_utils.return_distributed_iterators("gs://picard-testing-176520/BE_paper_pretraining/tfrecords",
                                 GLOBAL_BATCH_SIZE,
                                 196608,
                                 10,
                                 1536,
                                 args.num_heads,
                                 args.num_parallel,
                                 wandb.config.num_epochs,
                                 strategy,
                                 options)

                
            
            enformer_model = enformer.Enformer()
            SEQ_LENGTH = 196608

            options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
            checkpoint = tf.train.Checkpoint(module=enformer_model)#,options=options)
            tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
            latest = tf.train.latest_checkpoint("sonnet_weights")
            checkpoint.restore(latest,options=options).assert_existing_objects_matched()
            
    

            scheduler= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base,
                decay_steps=total_steps, alpha=(wandb.config.min_lr / wandb.config.lr_base))
            scheduler=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base,
                                         warmup_steps=wandb.config.warmup_frac*total_steps,
                                         decay_schedule_fn=scheduler)

            optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)

            metric_dict = {}
            
            freq_limit = int(wandb.config.input_length*wandb.config.freq_limit_scale)

            train_step, val_step, metric_dict = training_utils.return_train_val_functions(enformer_model,
                                                                                 optimizer,
                                                                                 strategy,
                                                                                 metric_dict, 
                                                                                 train_steps,
                                                                                 val_steps,
                                                                                 GLOBAL_BATCH_SIZE,
                                                                                 wandb.config.gradient_clip,
                                                                                 freq_limit=freq_limit,
                                                                                 fourier_loss_scale=wandb.config.fft_prior_scale) 
                
            
            ### main training loop
            global_step = 0
            val_losses = []
            val_pearsons = []
            val_R2 = []
            patience_counter = 0
            stop_criteria = False
            best_epoch = 0
            
            for epoch_i in range(1, wandb.config.num_epochs+1):
                print('starting epoch_', str(epoch_i))
                start = time.time()
                if epoch_i == 1:
                    # run once to build the model w/o updating anything
                    val_step(val_data)

                train_step(tr_data)
                    
                end = time.time()
                duration = (end - start) / 60.
                print('completed epoch ' + str(epoch_i))
                print('hg_train_loss: ' + str(metric_dict['hg_tr'].result().numpy()))
                wandb.log({'train_loss': metric_dict['hg_tr'].result().numpy()},
                          step=epoch_i)
                print('training duration(mins): ' + str(duration))
                
                start = time.time()
                val_step(val_data)
                
                print('val_loss: ' + str(metric_dict['hg_val'].result().numpy()))
                val_losses.append(metric_dict['hg_val'].result().numpy())
                wandb.log({'val_loss': metric_dict['hg_val'].result().numpy()},
                          step=epoch_i)
                print('pearsonsR: ')
                pearsonsR=metric_dict['pearsonsR'].result()['PearsonR'].numpy()
                print(pearsonsR)
                wandb.log({'rho(CD4_stim)': pearsonsR[0],
                           'rho(CD4_rest)': pearsonsR[1],
                           'rho(Jurkat_stim)': pearsonsR[2],
                           'rho(Jurkat_rest)': pearsonsR[3]},
                          step=epoch_i)
                
                val_pearsons.append(np.nanmedian(pearsonsR))
                print('R2: ')
                print(metric_dict['R2'].result()['R2'].numpy())
                

                end = time.time()
                duration = (end - start) / 60.
                print('completed epoch ' + str(epoch_i) + ' validation')
                print('validation duration(mins): ' + str(duration))
                print('patience counter at: ' + str(patience_counter))


                if (epoch_i > 2):
                    stop_criteria,patience_counter,best_epoch = \
                        training_utils.early_stopping(current_val_loss=val_losses[-1],
                                                        logged_val_losses=val_losses,
                                                        current_pearsons=val_pearsons[-1],
                                                        logged_pearsons=val_pearsons,
                                                        current_epoch=epoch_i,
                                                        best_epoch=best_epoch,
                                                        save_freq=args.savefreq,
                                                        patience=wandb.config.patience,
                                                        patience_counter=patience_counter,
                                                        min_delta=wandb.config.min_delta,
                                                        model=enformer_model,
                                                        save_directory=wandb.config.model_save_dir,
                                                        saved_model_basename=wandb.config.model_save_basename,
                                                        checkpoint=checkpoint)
                #plt.close('all')
                print('patience counter at: ' + str(patience_counter))
                for key, item in metric_dict.items():
                    item.reset_state()
                if stop_criteria:
                    print('early stopping at: epoch ' + str(epoch_i))
                    break
                    
            print('saving model at: epoch ' + str(epoch_i))
            print('best model was at: epoch ' + str(best_epoch))
            checkpoint.save(wandb.config.model_save_dir + "/" + wandb.config.model_save_basename + "_" + wandb.run.name + "/final/saved_model")
            #enformer_model.save_weights(wandb.config.model_save_dir + "/" + wandb.config.model_save_basename + "_" + wandb.run.name + "/final/saved_model")

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train)
    #sweep_train()

##########################################################################
if __name__ == '__main__':
    main()
        
