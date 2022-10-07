import tensorflow as tf
# Make sure the GPU is enabled
#assert tf.config.list_physical_devices('GPU'), 'Start the colab kernel with GPU: Runtime -> Change runtime type -> GPU'

import tensorflow_hub as hub
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import pybedtools as pybt
from deeplift import dinuc_shuffle

import sys

class Enformer:

    def __init__(self, tfhub_url):
        self._model = hub.load(tfhub_url).model


    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}

    @tf.function
    def contribution_input_grad(self, input_sequence,
                                target_mask, track_index,
                                output_head='human'):
        input_sequence = input_sequence[tf.newaxis]

        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)
            pred = self._model.predict_on_batch(input_sequence)[output_head][::track_index]
            #print(pred.shape)
            prediction = tf.reduce_sum(
                  target_mask[tf.newaxis] * pred) / target_mask_mass
        grad = tape.gradient(prediction, input_sequence)
        input_grad = grad * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)

        return tf.reduce_sum(input_grad, axis=-1), grad
    def vars_return(self):
        return self._model.signatures#['serving_default'].variables



# @title `variant_centered_sequences`
#with strategy.scope():
class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

def importance_scores(chrom, start, stop, target_index, mask_indices):

    target_interval = kipoiseq.Interval(chrom, int(start), int(stop))
    resized_interval = target_interval.resize(SEQUENCE_LENGTH)
    sequence_one_hot = one_hot_encode(fasta_extractor.extract(resized_interval))
    predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]

    target_mask = np.zeros_like(predictions)
    for idx in mask_indices:
        target_mask[idx, target_index] = 1
    # This will take some time since tf.function needs to get compiled.
    contribution_scores,grad = model.contribution_input_grad(sequence_one_hot.astype(np.float32), target_mask, target_index)
    contribution_scores = contribution_scores.numpy()
    pooled_contribution_scores = tf.nn.avg_pool1d(np.abs(contribution_scores)[np.newaxis,
                                                                              :, np.newaxis],
                                                  128, 128, 'VALID')[0, :, 0].numpy()

    base_scores = (sequence_one_hot[:][:].T * [contribution_scores[:],
                                                   contribution_scores[:],
                                                   contribution_scores[:],
                                                   contribution_scores[:]]).T

    gradient = np.multiply(sequence_one_hot[:][:].T, (np.squeeze(grad).T))
    ###### scrambled
    seq_shuffled = dinuc_shuffle.dinuc_shuffle(sequence_one_hot, 1)[0]
    
    target_mask = np.zeros_like(predictions)
    for idx in mask_indices:
        target_mask[idx, target_index] = 1
    # This will take some time since tf.function needs to get compiled.

    contribution_scores_scram, grad_scram = model.contribution_input_grad(seq_shuffled, target_mask, target_index)
    contribution_scores_scram = contribution_scores_scram.numpy()
    pooled_contribution_scores_scram = tf.nn.avg_pool1d(np.abs(contribution_scores_scram)[np.newaxis,
                                                                              :, np.newaxis],
                                                  128, 128, 'VALID')[0, :, 0].numpy()

    ## get base level matrix

    base_scores_scram = (seq_shuffled[:][:].T * [contribution_scores_scram[:],
                                        contribution_scores_scram[:],
                                        contribution_scores_scram[:],
                                        contribution_scores_scram[:]]).T

    ## get base level matri
    gradient_scram = np.multiply(seq_shuffled[:][:].T, (np.squeeze(grad_scram).T))

    return resized_interval,contribution_scores,pooled_contribution_scores,base_scores,np.squeeze(grad), sequence_one_hot,base_scores_scram

def write_out_bedgraph_pooled(pooled_contribution_scores, interval, filename_base):
    start = interval.start
    end =  interval.end
    chrom = interval.chrom
    name = '_'.join([str(chrom), str(start), str(end)])

    out_file = open(filename_base + '.pooled.bedGraph', 'w')


    for k, value in enumerate(pooled_contribution_scores):

        start_interval = k * 128 + start
        end_interval = (k+1) * 128 + start

        line = [str(chrom),
                str(start_interval), str(end_interval),
                str(value)]

        out_file.write('\t'.join(line) + '\n')
    out_file.close()

def write_out_bedgraph_all(contribution_scores, interval, filename_base):
    start = interval.start
    end =  interval.end
    chrom = interval.chrom
    name = '_'.join([str(chrom), str(start), str(end)])

    out_file = open(filename_base + '.all.bedGraph', 'w')


    for k, value in enumerate(contribution_scores):

        start_interval = start + k
        end_interval = start + k + 1

        line = [str(chrom),
                str(start_interval), str(end_interval),
                str(value)]

        out_file.write('\t'.join(line) + '\n')
    out_file.close()


def get_peak_scores(interval, peaks_file):
    start = interval.start + (128 * 1088) ### looking only at middle
    end =  interval.end - (128 * 1088)

    chrom = interval.chrom
    interval_str = '\t'.join([str(chrom), str(start), str(end)])
    interval_bed = pybt.BedTool(interval_str, from_string=True)

    peaks = pybt.BedTool(peaks_file)

    intersection = peaks.intersect(interval_bed)

    to_extract = []

    intersection_intervals = intersection.to_dataframe()

    for index, row in intersection_intervals.iterrows():
        interval_middle = (int(row['start']) + int(row['end'])) // 2
        
        if ((interval_middle + 750 < int(start))):
            continue
        if ((interval_middle - 750 > int(end))):
            continue
        
        if interval_middle - 750 < int(start):
            index_start = 0
            index_end = 750
        elif interval_middle + 750 > int(end):
            index_end = int(end) - int(start)
            index_start = index_end - 750
        else:
            index_start = interval_middle - 750 - int(start)
            index_end = interval_middle + 750 - int(start)
        
        to_extract.append((index_start,index_end))

    return to_extract

model_path = 'https://tfhub.dev/deepmind/enformer/1'
fasta_file = sys.argv[1]

SEQUENCE_LENGTH = 393216

model = Enformer(model_path)
fasta_extractor = FastaStringExtractor(fasta_file)
chrom, start, stop = sys.argv[2].split(',')
mask_indices = [int(k) for k in sys.argv[3].split(',')]
target_index = sys.argv[4]
out_name = sys.argv[5]
peaks = sys.argv[6]

resized_interval, contribution_scores, pooled_contribution_scores, base_scores,gradient, sequence_one_hot,base_scores_scram = importance_scores(chrom, int(start), int(stop),int(target_index),mask_indices)

### get peaks
to_extract = get_peak_scores(resized_interval, peaks)

base_scores_subset = []
hyp_subset = []
seq_subset = []
null_subset = []

for item in to_extract:
    base_scores_sub = np.array(base_scores[item[0]-1:item[1]-1, :])
    
    if base_scores_sub.sum(axis=1).sum(axis=0) == 0:
        continue
    
    base_scores_subset.append(base_scores[item[0]-1:item[1]-1, :])
    hyp_subset.append(gradient[item[0]-1:item[1]-1, :])
    seq_subset.append(sequence_one_hot[item[0]-1:item[1]-1, :])
    null_subset.append(base_scores_scram[item[0]-1:item[1]-1, :])

write_out_bedgraph_pooled(pooled_contribution_scores, resized_interval, out_name)
write_out_bedgraph_all(contribution_scores, resized_interval, out_name)


np.savetxt(out_name + ".basescores.out", base_scores, fmt='%10.8f')
np.savetxt(out_name + ".gradient.out", gradient, fmt='%10.8f')
np.savetxt(out_name + ".sequence", sequence_one_hot, fmt='%s')

np.save(out_name + ".peaks.basescores.npy", np.array(base_scores_subset))
np.save(out_name + ".peaks.gradient.npy", np.array(hyp_subset))
np.save(out_name + ".peaks.null.npy", np.array(null_subset))
np.save(out_name + ".peaks.sequence.npy", np.array(seq_subset))