dfrom __future__ import print_function, division

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3
        
        
import numpy as np
import modisco
import sys
import os
import glob


"""
script that computes TFModisco motif clusters from Enformer base importance scores
across a range of tasks. specify:
 - basescore_files(input * gradient)
 - gradient_files(gradient * 4 x N matrix of ones) as hypothetical importance scores
 - sequence files - sequence corresponding to above that will be one hot encoded
 - null files - basescores but for scrambled sequences(dinucleotide shuffled)
"""

basescore_files = sorted(glob.glob("~/enformer_results/basescores/*.peaks.basescores.npy"))
grad_files = sorted(glob.glob("~/enformer_results/gradients/*.peaks.gradient.npy"))
sequence_files = sorted(glob.glob("~/enformer_results/sequence/*.peaks.sequence.npy"))
null_files = sorted(glob.glob("~/enformer_results/basescores_scramble/*.peaks.null.npy"))

task_names = []
basescores = []
hyp_tasks = []
sequences = []
nulls = []


for k,file in enumerate(basescore_files):
    task_name = file.split('/')[-1].split('.')[0]
    task_names.append(task_name)
    #if task_name not in lst:
    #    continue
    
    basescore_arr = np.load(file,allow_pickle=True)
    if len(basescore_arr.shape) != 3:
        continue
    if basescore_arr.shape[1] != 1500:
        continue 
    grad_arr = np.load(grad_files[k],allow_pickle=True)
    seq_arr = np.load(sequence_files[k],allow_pickle=True)
    null_arr = np.load(null_files[k],allow_pickle=True)

    for j, entry in enumerate(basescore_arr):
        basescores.append(entry[375:1125])
        hyp_tasks.append(grad_arr[j][375:1125])
        sequences.append(seq_arr[j][375:1125])
        nulls.append(null_arr[j][375:1125])
            
#nulldist_perposimp = np.sum([x for x in nulls if len(x) == 3000], axis=-1)
nulldist_perposimp = np.asarray([np.sum(x,axis=-1) for x in nulls]) / len(nulls)


import h5py
import numpy as np
import modisco
import random

#Uncomment to refresh modules for when tweaking code during development:
from importlib import reload
reload(modisco.util)
reload(modisco.pattern_filterer)
reload(modisco.aggregator)
reload(modisco.core)
reload(modisco.seqlet_embedding.advanced_gapped_kmer)
reload(modisco.affinitymat.transformers)
reload(modisco.affinitymat.core)
reload(modisco.affinitymat)
reload(modisco.cluster.core)
reload(modisco.cluster)
reload(modisco.tfmodisco_workflow.seqlets_to_patterns)
reload(modisco.tfmodisco_workflow)
reload(modisco)
from deeplift import dinuc_shuffle

tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
                    #Slight modifications from the default settings
                    sliding_window_size=15,
                    flank_size=5,
                    target_seqlet_fdr=0.15,
                    seqlets_to_patterns_factory=
                     modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                        #Note: as of version 0.5.6.0, it's possible to use the results of a motif discovery
                        # software like MEME to improve the TF-MoDISco clustering. To use the meme-based
                        # initialization, you would specify the initclusterer_factory as shown in the
                        # commented-out code below:
                        #initclusterer_factory=modisco.clusterinit.memeinit.MemeInitClustererFactory(    
                        #    meme_command="meme", base_outdir="meme_out",            
                        #    max_num_seqlets_to_use=10000, nmotifs=10, n_jobs=1),
                        trim_to_window_size=15,
                        initial_flank_to_add=5,
                        final_flank_to_add=5,
                        final_min_cluster_size=60,
                        #use_pynnd=True can be used for faster nn comp at coarse grained step
                        # (it will use pynndescent), but note that pynndescent may crash
                        #use_pynnd=True, 
                        n_cores=32)
                )(
                task_names=["task0"],
                contrib_scores={'task0': basescores},                
                hypothetical_contribs={'task0': hyp_tasks},
                one_hot=sequences,
                null_per_pos_scores={'task0':nulldist_perposimp})



import h5py
import modisco.util
#reload(modisco.util)
#![[ -e results.hdf5 ]] && rm results.hdf5
grp = h5py.File("results_DE.hdf5", "w")
tfmodisco_results.save_hdf5(grp)
grp.close()