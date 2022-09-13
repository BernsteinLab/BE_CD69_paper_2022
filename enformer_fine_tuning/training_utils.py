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
from datetime import datetime
import random

import multiprocessing
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

import pandas as pd
import seaborn as sns

from scipy.stats.stats import pearsonr, spearmanr
from scipy.stats import linregress
from scipy import stats
import keras.backend as kb

import scipy.special
import scipy.stats
import scipy.ndimage

import metrics

tf.keras.backend.set_floatx('float32')

def tf_tpu_initialize(tpu_name):
    """Initialize TPU and return global batch size for loss calculation
    Args:
        tpu_name
    Returns:
        distributed strategy
    """
    
    try: 
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_name)
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)

    except ValueError: # no TPU found, detect GPUs
        strategy = tf.distribute.get_strategy()

    return strategy


"""
having trouble w/ providing organism/step inputs to train/val steps w/o
triggering retracing/metadata resource exhausted errors, so defining 
them separately for hg, mm 
to do: simplify to two functions w/ organism + mini_batch_step_inputs
consolidate into single simpler function
"""


def return_train_val_functions(model,
                               optimizer,
                               strategy,
                               metric_dict,
                               train_steps, 
                               val_steps,
                               global_batch_size,
                               gradient_clip,
                               freq_limit=5000,
                               fourier_loss_scale=1.0):
    """Returns distributed train and validation functions for
    a given list of organisms
    Args:
        model: model object
        optimizer: optimizer object
        metric_dict: empty dictionary to populate with organism
                     specific metrics
        train_steps: number of train steps to take in single epoch
        val_steps: number of val steps to take in single epoch
        global_batch_size: # replicas * batch_size_per_replica
        gradient_clip: gradient clip value to be applied in case of adam/adamw optimizer
    Returns:
        distributed train function
        distributed val function
        metric_dict: dict of tr_loss,val_loss, correlation_stats metrics
                     for input organisms
    
    return distributed train and val step functions for given organism
    train_steps is the # steps in a single epoch
    val_steps is the # steps to fully iterate over validation set
    """


    metric_dict["hg_tr"] = tf.keras.metrics.Mean("hg_tr_loss",
                                                 dtype=tf.float32)
    metric_dict["hg_val"] = tf.keras.metrics.Mean("hg_val_loss",
                                                  dtype=tf.float32)
    
    metric_dict['pearsonsR'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    
    metric_dict['R2'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})

    def dist_train_step_transfer(iterator):
        @tf.function(jit_compile=True)
        def train_step(inputs):
            target=tf.cast(inputs['target'],
                           dtype = tf.float32)
            sequence=tf.cast(inputs['sequence'],
                             dtype=tf.float32)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                with tf.GradientTape(watch_accessed_variables=False) as input_grad_tape:
                    input_grad_tape.watch(sequence)
                    #heads_vars = [v for v in model.trainable_variables if "heads" in v.name]
                    all_vars = model.trainable_variables
                    #print('all_vars')
                    #print(all_vars)
                    heads_vars = [x for x in all_vars if "new_head" in x.name]
                    #print('heads_vars')
                    #print(heads_vars)
                    for var in heads_vars:
                        tape.watch(var)
                    stem_vars = [x for x in all_vars if "stem" in x.name]
                    #print('stem_vars')
                    #print(stem_vars)
                    for var in stem_vars:
                        tape.watch(var)
                    vars_subset = [x for x in all_vars if (("new_head" in x.name) or ("stem" in x.name))]
                    #print('vars_subset')
                    #print(vars_subset)
                    output = model(sequence, is_training=True)
                #print(output)
                input_grads = input_grad_tape.gradient(output, 
                                                       sequence)
                
                loss = tf.reduce_sum(poisson(target,
                                             output)) * (1. / global_batch_size)
                #print(input_grads)
                #if use_prior and fourier_loss_scale is not None:
                input_grads = input_grads * sequence
                #print(input_grads)
                fourier_loss = fourier_att_prior_loss(input_grads) * fourier_loss_scale
                loss = loss + fourier_loss
                
            gradients = tape.gradient(loss, vars_subset)
                          #*model.trunk.submodules[0].trainable_variables])
            #gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip) 
            optimizer.apply_gradients(zip(gradients, heads_vars))
            
            metric_dict["hg_tr"].update_state(loss)

        for _ in tf.range(train_steps): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(train_step, args=(next(iterator),))
            
    def dist_val_step(iterator): #input_batch, model, optimizer, organism, gradient_clip):
        @tf.function(jit_compile=True)
        def val_step(inputs):
            target=tf.cast(inputs['target'],
                           dtype = tf.float32)
            sequence=tf.cast(inputs['sequence'],
                             dtype=tf.float32)
            output = model(sequence, is_training=False)
            loss = tf.reduce_sum(poisson(target,
                                         output)) * (1. / global_batch_size)
            metric_dict["hg_val"].update_state(loss)
            metric_dict['pearsonsR'].update_state(target, output)
            metric_dict['R2'].update_state(target, output)

        for _ in tf.range(val_steps): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(val_step, args=(next(iterator),))
        
    return dist_train_step_transfer, dist_val_step, metric_dict


def deserialize_tr(serialized_example,input_length,max_shift, out_length,num_targets):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string),
    }
    
    data = tf.io.parse_example(serialized_example, feature_map)

    shift = random.randrange(0,max_shift)
    input_seq_length = input_length + max_shift
    interval_end = input_length + shift
    
    ### rev_comp
    rev_comp = random.randrange(0,2)

    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (input_length + max_shift, 4))
    sequence = tf.cast(sequence, tf.float32)
    sequence = tf.slice(sequence, [shift,0],[input_length,-1])
    
    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (out_length, num_targets))
    target = tf.slice(target,
                      [320,0],
                      [896,-1])
    
    if rev_comp == 1:
        sequence = tf.gather(sequence, [3, 2, 1, 0], axis=-1)
        sequence = tf.reverse(sequence, axis=[0])
        target = tf.reverse(target,axis=[0])
    
    return {'sequence': tf.ensure_shape(sequence,
                                        [input_length,4]),
            'target': tf.ensure_shape(target,
                                      [896,num_targets])}
                    


def deserialize_val(serialized_example,input_length,max_shift, out_length,num_targets):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string),
    }
    
    data = tf.io.parse_example(serialized_example, feature_map)

    shift = 5
    input_seq_length = input_length + max_shift
    interval_end = input_length + shift
    
    ### rev_comp
    rev_comp = random.randrange(0,2)

    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (input_length + max_shift, 4))
    sequence = tf.cast(sequence, tf.float32)
    sequence = tf.slice(sequence, [shift,0],[input_length,-1])
    
    if rev_comp == 1:
        sequence = tf.gather(sequence, [3, 2, 1, 0], axis=-1)
        sequence = tf.reverse(sequence, axis=[0])
        target = tf.reverse(target,axis=[0])
    
    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (out_length, num_targets))
    target = tf.slice(target,
                      [320,0],
                      [896,-1])
    
    return {'sequence': tf.ensure_shape(sequence,
                                        [input_length,4]),
            'target': tf.ensure_shape(target,
                                      [896,num_targets])}
                    
def return_dataset(gcs_path,
                   split,
                   batch,
                   input_length,
                   max_shift,
                   out_length,
                   num_targets,
                   options,
                   num_parallel,
                   num_epoch):
    """
    return a tf dataset object for given gcs path
    """
    wc = str(split) + "*.tfr"
    
    list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                wc)))

    files = tf.data.Dataset.list_files(list_files)
    
    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)
    if split == 'train':
        dataset = dataset.map(lambda record: deserialize_tr(record,
                                                         input_length,
                                                         max_shift,
                                                         out_length,
                                                         num_targets),
                              deterministic=False,
                              num_parallel_calls=num_parallel)
        
    else:
        dataset = dataset.map(lambda record: deserialize_val(record,
                                                         input_length,
                                                         max_shift,
                                                         out_length,
                                                         num_targets),
                              deterministic=False,
                              num_parallel_calls=num_parallel)

    return dataset.repeat(num_epoch).batch(batch,drop_remainder=True).prefetch(1)



def return_distributed_iterators(gcs_path,
                                 global_batch_size,
                                 input_length,
                                 max_shift,
                                 out_length,
                                 num_targets,
                                 num_parallel_calls,
                                 num_epoch,
                                 strategy,
                                 options):
    """ 
    returns train + val dictionaries of distributed iterators
    for given heads_dictionary
    """
    with strategy.scope():
        tr_data = return_dataset(gcs_path,
                                 "train",
                                 global_batch_size,
                                 input_length,
                                 max_shift,
                                 out_length,
                                 num_targets,
                                 options,
                                 num_parallel_calls,
                                 num_epoch)
        
        
            
        val_data = return_dataset(gcs_path,
                                 "val",
                                 global_batch_size,
                                 input_length,
                                 max_shift,
                                 out_length,
                                 num_targets,
                                 options,
                                 num_parallel_calls,
                                 num_epoch)
            
            
        train_dist = strategy.experimental_distribute_dataset(tr_data)
        val_dist= strategy.experimental_distribute_dataset(val_data)

        tr_data_it = iter(train_dist)
        val_data_it = iter(val_dist)


    return tr_data_it, val_data_it



def early_stopping(current_val_loss,
                   logged_val_losses,
                   current_pearsons,
                   logged_pearsons,
                   current_epoch,
                   best_epoch,
                   save_freq,
                   patience,
                   patience_counter,
                   min_delta,
                   model,
                   save_directory,
                   saved_model_basename,
                   checkpoint):
    """early stopping function
    Args:
        current_val_loss: current epoch val loss
        logged_val_losses: previous epochs val losses
        current_epoch: current epoch number
        save_freq: frequency(in epochs) with which to save checkpoints
        patience: # of epochs to continue w/ stable/increasing val loss
                  before terminating training loop
        patience_counter: # of epochs over which val loss hasn't decreased
        min_delta: minimum decrease in val loss required to reset patience 
                   counter
        model: model object
        save_directory: cloud bucket location to save model
        model_parameters: log file of all model parameters 
        saved_model_basename: prefix for saved model dir
    Returns:
        stop_criteria: bool indicating whether to exit train loop
        patience_counter: # of epochs over which val loss hasn't decreased
        best_epoch: best epoch so far 
    """
    ### check if min_delta satisfied
    try: 
        best_loss = min(logged_val_losses[:-1])
        best_pearsons=max(logged_pearsons[:-1])
        
    except ValueError:
        best_loss = current_val_loss
        best_pearsons = current_pearsons
        
    stop_criteria = False
    ## if min delta satisfied then log loss
    
    if (current_val_loss >= (best_loss - min_delta)):# and (current_pearsons <= best_pearsons):
        patience_counter += 1
        if patience_counter >= patience:
            stop_criteria=True
    else:

        best_epoch = np.argmin(logged_val_losses)
        ## save current model
        if (current_epoch % save_freq) == 0:
            print('Saving model...')
            model_name = save_directory + "/" + \
                            saved_model_basename + "/iteration_" + \
                                str(current_epoch) + "/checkpoint"
            #model.save_weights(model_name)
            checkpoint.save(model_name)
            ### write to logging file in saved model dir to model parameters and current epoch info
            
        patience_counter = 0
        stop_criteria = False
    
    return stop_criteria, patience_counter, best_epoch
        
        
def parse_args(parser):
    """Loads in command line arguments
    """
        
    parser.add_argument('--tpu_name', dest = 'tpu_name',
                        help='tpu_name')
    parser.add_argument('--tpu_zone', dest = 'tpu_zone',
                        help='tpu_zone')
    parser.add_argument('--wandb_project', 
                        dest='wandb_project',
                        help ='wandb_project')
    parser.add_argument('--wandb_user',
                        dest='wandb_user',
                        help ='wandb_user')
    parser.add_argument('--wandb_sweep_name',
                        dest='wandb_sweep_name',
                        help ='wandb_sweep_name')
    parser.add_argument('--gcs_project', dest = 'gcs_project',
                        help='gcs_project')
    parser.add_argument('--gcs_path',
                        dest='gcs_path',
                        help= 'google bucket containing preprocessed data')
    parser.add_argument('--num_parallel', dest = 'num_parallel',
                        type=int, default=tf.data.AUTOTUNE,
                        help='thread count for tensorflow record loading')
    parser.add_argument('--batch_size', dest = 'batch_size',
                        type=int, help='batch_size')
    parser.add_argument('--num_epochs', dest = 'num_epochs',
                        type=int, help='num_epochs')
    parser.add_argument('--warmup_frac', dest = 'warmup_frac',
                        default=0.0,
                        type=float, help='warmup_frac')
    parser.add_argument('--patience', dest = 'patience',
                        type=int, help='patience for early stopping')
    parser.add_argument('--min_delta', dest = 'min_delta',
                        type=float, help='min_delta for early stopping')
    parser.add_argument('--model_save_dir',
                        dest='model_save_dir',
                        type=str)
    parser.add_argument('--model_save_basename',
                        dest='model_save_basename',
                        type=str)
    parser.add_argument('--input_length',
                        dest='input_length',
                        default=196608,
                        type=int,
                        help='input_length')
    parser.add_argument('--lr_base',
                        dest='lr_base',
                        default="1.0e-03",
                        help='lr_base')
    parser.add_argument('--min_lr',
                        dest='min_lr',
                        default="1.0e-07",
                        help= 'min_lr')
    parser.add_argument('--epsilon',
                        dest='epsilon',
                        default=1.0e-10,
                        type=float,
                        help= 'epsilon')
    parser.add_argument('--rectify',
                        dest='rectify',
                        default=True,
                        help= 'rectify')
    parser.add_argument('--gradient_clip',
                        dest='gradient_clip',
                        type=str,
                        default="0.2",
                        help= 'gradient_clip')
    parser.add_argument('--weight_decay_frac',
                        dest='weight_decay_frac',
                        type=str,
                        help= 'weight_decay_frac')
    parser.add_argument('--sync_period',
                        type=int,
                        dest='sync_period',
                        help= 'sync_period')
    parser.add_argument('--slow_step_frac',
                        type=float,
                        dest='slow_step_frac',
                        help= 'slow_step_frac')
    parser.add_argument('--num_heads',
                        dest='num_heads',
                        type=int,
                        help= 'num_heads')
    parser.add_argument('--savefreq',
                        dest='savefreq',
                        type=int,
                        help= 'savefreq')
    parser.add_argument('--use_fft_prior',
                        dest='use_fft_prior',
                        default="True",
                        help= 'use_fft_prior')
    parser.add_argument('--freq_limit_scale',
                        dest='freq_limit_scale',
                        type=str,
                        default="0.07",
                        help= 'freq_limit')
    parser.add_argument('--fft_prior_scale',
                        dest='fft_prior_scale',
                        type=str,
                        default="0.5",
                        help= 'fft_prior_scale')
    parser.add_argument('--total_steps',
                        dest='total_steps',
                        type=int,
                        default=0,
                        help= 'total_steps')
    parser.add_argument('--beta1',
                        dest='beta1',
                        type=str,
                        default="0.90",
                        help= 'beta1')
    parser.add_argument('--beta2',
                        dest='beta2',
                        type=str,
                        default="0.999",
                        help= 'beta2')
    
    args = parser.parse_args()
    return parser
    
    
    
def one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    vocabulary = tf.constant(['A', 'C', 'G', 'T'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=0)

    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))

    out = tf.one_hot(table.lookup(input_characters), 
                      depth = 4, 
                      dtype=tf.float32)
    return out

def rev_comp_one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))
    input_characters = tf.reverse(input_characters,[0])
    
    vocabulary = tf.constant(['T', 'G', 'C', 'A'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=0)

    out = tf.one_hot(table.lookup(input_characters), 
                      depth = 4, 
                      dtype=tf.float32)
    return out



def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


def fourier_att_prior_loss(
    input_grads, freq_limit=5000, limit_softness=0.2,
    att_prior_grad_smooth_sigma=3):
    """
    Computes an attribution prior loss for some given training examples,
    using a Fourier transform form.
    Arguments:
        `output`: a B-tensor, where B is the batch size; each entry is a
            predicted logTPM value
        `input_grads`: a B x L x 4 tensor, where B is the batch size, L is
            the length of the input; this needs to be the gradients of the
            input with respect to the output; this should be
            *gradient times input*
        `freq_limit`: the maximum integer frequency index, k, to consider for
            the loss; this corresponds to a frequency cut-off of pi * k / L;
            k should be less than L / 2
        `limit_softness`: amount to soften the limit by, using a hill
            function; None means no softness
        `att_prior_grad_smooth_sigma`: amount to smooth the gradient before
            computing the loss
    Returns a single scalar Tensor consisting of the attribution loss for
    the batch.
    """
    abs_grads = kb.sum(kb.abs(input_grads), axis=2)

    # Smooth the gradients
    grads_smooth = smooth_tensor_1d(
        abs_grads, att_prior_grad_smooth_sigma
    )
    
    # Only do the positives
    #pos_grads = grads_smooth[status == 1]

    #if pos_grads.numpy().size:
    pos_fft = tf.signal.rfft(tf.cast(abs_grads,dtype=tf.float32))
    pos_mags = tf.abs(pos_fft)
    pos_mag_sum = kb.sum(pos_mags, axis=1, keepdims=True)
    zero_mask = tf.cast(pos_mag_sum == 0, tf.float32)
    pos_mag_sum = pos_mag_sum + zero_mask  # Keep 0s when the sum is 0  
    pos_mags = pos_mags / pos_mag_sum

    # Cut off DC
    pos_mags = pos_mags[:, 1:]

    # Construct weight vector
    if limit_softness is None:
        weights = tf.sequence_mask(
            [freq_limit], maxlen=tf.shape(pos_mags)[1], dtype=tf.float32
        )
    else:
        weights = tf.sequence_mask(
            [freq_limit], maxlen=tf.shape(pos_mags)[1], dtype=tf.float32
        )
        x = tf.abs(tf.range(
            -freq_limit + 1, tf.shape(pos_mags)[1] - freq_limit + 1, dtype=tf.float32
        ))  # Take absolute value of negatives just to avoid NaN; they'll be removed
        decay = 1 / (1 + tf.pow(x, limit_softness))
        weights = weights + ((1.0 - weights) * decay)

    # Multiply frequency magnitudes by weights
    pos_weighted_mags = pos_mags * weights

    # Add up along frequency axis to get score
    pos_score = tf.reduce_sum(pos_weighted_mags, axis=1)
    pos_loss = 1 - pos_score
    return tf.reduce_mean(pos_loss)
    
    
    
    
def smooth_tensor_1d(input_tensor, smooth_sigma):
    """
    Smooths an input tensor along a dimension using a Gaussian filter.
    Arguments:
        `input_tensor`: a A x B tensor to smooth along the second dimension
        `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
            standard deviation of the Gaussian to use, and the Gaussian will be
            truncated after 1 sigma (i.e. the smoothing window is
            1 + (2 * sigma); sigma of 0 means no smoothing
    Returns an array the same shape as the input tensor, with the dimension of
    `B` smoothed.
    """
    input_tensor = tf.cast(input_tensor,dtype=tf.float32)
    # Generate the kernel
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1
    base = np.zeros(1 + (2 * sigma))
    base[sigma] = 1  # Center of window is 1 everywhere else is 0
    kernel = scipy.ndimage.gaussian_filter(base, 
                                           sigma=sigma, 
                                           truncate=truncate)
    kernel = tf.constant(kernel,dtype=tf.float32)

    # Expand the input and kernel to 3D, with channels of 1
    input_tensor = tf.expand_dims(input_tensor, axis=2)  # Shape: A x B x 1
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=1), axis=2)  # Shape: (1 + 2s) x 1 x 1

    smoothed = tf.nn.conv1d(
        input_tensor, kernel, stride=1, padding="SAME", data_format="NWC"
    )

    return tf.squeeze(smoothed, axis=2)

def poisson(y_true, y_pred):

    
    return tf.reduce_mean(y_pred - (y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())),
                                            axis=-1)
    