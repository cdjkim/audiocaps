from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import os
__PATH__ = os.path.abspath(os.path.dirname(__file__))

import tensorflow as tf
import numpy as np
import h5py
import colorlog

from utils import vocab_utils, etc_utils


class AudioCapsInput(
    namedtuple("AudioCapsInput",
               ("answer", "answer_length", "audio_feature", "audio_length", "labels", "num_labels",
                "access_id"))):
    pass


class fc3c4Pyramid2TextInput(
    namedtuple("AudioCapsInput",
               ("answer", "answer_length", "feature_FC3", "feature_C4", "FC3_length","C4_length","labels",
                "num_labels",  "access_id"))):
    pass


def _get_feature_shape(feature_name, data_name):
    if feature_name == "fc3c4_pyramid":
        modal_shape = [[10,128],[10, 12, 8, 512]]
    else:
        raise ValueError("{} is not supported yet".format(feature_name))
    return modal_shape


def prefetch_dataset(batch_size, bucket_width, buffer_size, random_seed,
                     num_gpus, num_epochs, mode, feature_name):

    data_name = 'audiocaps'
    base_data_dir = os.path.join(__PATH__, '..', 'data/audiocaps')

    # Read base text file
    data_dir = os.path.join(base_data_dir)
    data_fname = os.path.join(data_dir, 'features/auxiliary', mode + '.txt')

    if mode in ['val', 'test']:
        dataset = tf.data.TextLineDataset(data_fname).repeat(1)

    else:
        dataset = tf.data.TextLineDataset(data_fname).shuffle(buffer_size).repeat(num_epochs)

    # Read feature file
    if feature_name == "fc3c4_pyramid":
        feature_fname_FC3 = os.path.join(base_data_dir, 'features', 'vggish_last_pad.hdf5')
        feature_fp_FC3 = h5py.File(feature_fname_FC3, 'r')
        feature_fname_C4 = os.path.join(base_data_dir, 'features', 'vggish_conv4_pad.hdf5')
        feature_fp_C4 = h5py.File(feature_fname_C4, 'r')
    else:
        feature_fname = os.path.join(base_data_dir, 'features', '{}.hdf5'.format(feature_name))
        feature_fp = h5py.File(feature_fname, 'r')

    # NOTE : modify here if new feature is added
    modal_shape = _get_feature_shape(feature_name, data_name)

    def _read_modal_fn(modal_id: bytes):

        if feature_name == "fc3c4_pyramid":
            # modal_shape = [Time, 128]
            FC3_feature = feature_fp_FC3[modal_id].value.astype(np.float32)
            C4_feature = feature_fp_C4[modal_id].value.astype(np.float32)
            return FC3_feature,  C4_feature, np.int32(FC3_feature.shape[0]),np.int32(C4_feature.shape[0])

        else:
            audio_feature = feature_fp[modal_id].value.astype(np.float32)
            return audio_feature, np.int32(audio_feature.shape[0])

    def _parse_fn(text_line):
        if data_name == "audiocaps":
            access_id, answer, labels = tf.decode_csv(text_line, [[""], [""], [""]], ",")
        else:
            raise ValueError("{} is not supported".format(data_name))

        answer = tf.concat(([vocab_utils.GO_ID], _list_split(answer), [vocab_utils.EOS_ID]), 0)
        answer_length = tf.size(answer)

        labels = _list_split(labels)
        num_labels = tf.size(labels)

        if feature_name == "fc3c4_pyramid":
            FC3_feature, C4_feature, FC3_length, C4_length \
                = tf.py_func(_read_modal_fn, [access_id], [tf.float32, tf.float32, tf.int32, tf.int32])
            FC3_feature.set_shape(modal_shape[0])
            C4_feature.set_shape(modal_shape[1])

            return [answer, answer_length, FC3_feature, C4_feature, FC3_length, C4_length, labels, num_labels, access_id]

        else:
            modal_feature, modal_length = tf.py_func(_read_modal_fn, [access_id], [tf.float32, tf.int32])
            modal_feature.set_shape(modal_shape)
            return [answer, answer_length, modal_feature, modal_length, labels, num_labels, access_id]

    dataset = dataset.map(_parse_fn, num_parallel_calls=30)

    if feature_name == "fc3c4_pyramid":
        batched_dataset = _bucketing(dataset, 1, batch_size, bucket_width, mode,
                                    padded_shapes=(tf.TensorShape([None]),
                                                    tf.TensorShape([]),
                                                    tf.TensorShape(modal_shape[0]),
                                                    tf.TensorShape(modal_shape[1]),
                                                    tf.TensorShape([]),
                                                    tf.TensorShape([]),
                                                    tf.TensorShape([None]),
                                                    tf.TensorShape([]),
                                                    tf.TensorShape([])))
    else:
        batched_dataset = _bucketing(dataset, 1, batch_size, bucket_width, mode,
                                    padded_shapes=(tf.TensorShape([None]),
                                                    tf.TensorShape([]),
                                                    tf.TensorShape(modal_shape),
                                                    tf.TensorShape([]),
                                                    tf.TensorShape([None]),
                                                    tf.TensorShape([]),
                                                    tf.TensorShape([])))

    batched_iter = batched_dataset.make_initializable_iterator()
    batched_inputs = []
    for i in range(num_gpus):
        features = batched_iter.get_next()
        if feature_name == "fc3c4_pyramid":
            batched_input = fc3c4Pyramid2TextInput(answer=features[0],
                                            answer_length=features[1],
                                            feature_FC3=features[2],
                                            feature_C4=features[3],
                                            FC3_length=features[4],
                                            C4_length=features[5],
                                            labels=features[6],
                                            num_labels=features[7],
                                            access_id=features[8])

        else:
            batched_input = AudioCapsInput(answer=features[0],
                                            answer_length=features[1],
                                            audio_feature=features[2],
                                            audio_length=features[3],
                                            labels=features[4],
                                            num_labels=features[5],
                                            access_id=features[6])

        batched_inputs.append(batched_input)

    num_fname = os.path.join(data_dir, "features", "auxiliary", "num_%s.txt" % mode)

    with open(num_fname, 'r') as f:
        num_lines = int(f.readline())

    return num_lines, batched_iter.initializer, batched_inputs


def _list_split(line, delimiter=' '):
    return tf.string_to_number(tf.string_split([line], delimiter).values, tf.int32)


def _bucketing(dataset, key_idx, batch_size, bucket_width, mode, padded_shapes=None):
    """creates a batch accessible by key.
    """
    def _batching_func(x):
        # automatically pads the elements in the batch.
        return x.padded_batch(batch_size, padded_shapes=padded_shapes)

    if 'train' in mode:
        def _key_func(*args):
            bucket_id = args[key_idx] / bucket_width
            return tf.to_int64(bucket_id)

        def _reduce_func(unused_key, windowed_data):
            return _batching_func(windowed_data)

        batched_dataset = dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=_key_func,
                reduce_func=_reduce_func,
                window_size=batch_size
            )
        )

    elif 'test' in mode:
        batched_dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(batch_size, padded_shapes))

    return batched_dataset
