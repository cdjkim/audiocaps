# for set_logger
import logging
import colorlog

# for sort_dict
import operator

# for timestamp
import calendar
import datetime

import os

import numpy as np
from utils import vocab_utils



def set_logger(fname=None):
    colorlog.basicConfig(
        filename=fname,
        level=logging.INFO,
        format="%(log_color)s[%(levelname)s:%(asctime)s]%(reset)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def timestamp_to_utc(timestamp):
    return datetime.datetime.utcfromtimestamp(timestamp)


def utc_to_timestamp(utc):
    return calendar.timegm(utc.utctimetuple())


def split_list(in_list, num_splits):
    avg_len = len(in_list) / float(num_splits)
    out_list = []
    last_idx = 0.0
    while last_idx < len(in_list):
        out_list.append(in_list[int(last_idx):int(last_idx + avg_len)])
        last_idx += avg_len
    return out_list


def sort_dict(dic):
    # Sort by alphabet
    sorted_pair_list = sorted(dic.items(), key=operator.itemgetter(0))
    # Sort by count
    sorted_pair_list = sorted(sorted_pair_list, key=operator.itemgetter(1), reverse=True)
    return sorted_pair_list


def dict_to_matrix(dictionary):
    return [[k, str(w)] for k, w in sorted(dictionary.items())]


def _save_prediction_answer(predictions, answers, access_ids, save_path, global_step):
    pred_fp = open(os.path.join(save_path, 'pred_{}.txt'.format(global_step)), 'w')
    answer_fp = open(os.path.join(save_path, 'answer_{}.txt'.format(global_step)), 'w')
    access_id_fp = open(os.path.join(save_path, 'access_id_{}.txt'.format(global_step)), 'w')
    for pred, answer, access_id in zip(predictions, answers, access_ids):
        pred_fp.write(pred + '\n')
        answer_fp.write(answer + '\n')
        access_id_fp.write(access_id + '\n')
    pred_fp.close()
    answer_fp.close()
    access_id_fp.close()


def _trim_after_eos(sentences):
    trimmed_sentences = []
    for sentence in sentences:
        try:
            eos_idx = int(np.where(sentence == vocab_utils._EOS)[0][0])
            trimmed_sentence = ' '.join(sentence[:eos_idx])
        except IndexError:
            trimmed_sentence = ' '.join(sentence)
        trimmed_sentences.append(trimmed_sentence)
    return trimmed_sentences


