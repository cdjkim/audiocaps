import time

import os
import pickle
import tensorflow as tf
import numpy as np
import colorlog

from helpers.nlp_helper import load_fasttext, load_cached_vector


def dict_to_summary(dictionary):
    summary = tf.Summary()
    for key, value in dictionary.items():
        summary.value.add(tag=key, simple_value=value)
    return summary


def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    colorlog.info(
        "Loaded %s model from %s, time %.2fs" % (
            name, ckpt, time.time() - start_time
        )
    )
    return model


def create_or_load_model(model, model_dir, session, name, specific_ckpt=None):
    """Create model and initialize or load parameters in session."""

    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if specific_ckpt:
        specific_ckpt_path = os.path.join(model_dir, specific_ckpt)
        model = load_model(model, specific_ckpt_path, session, name)
        fresh = False
    elif latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
        fresh = False
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        colorlog.info("  created %s model with new parameters, time %.2fs" %
                        (name, time.time() - start_time))
        fresh = True

    return model, fresh


def clip_grads(grads, max_grad_norm):
    if max_grad_norm <= 0.0:
        return grads
    else:
        with tf.name_scope('gradient_clipping'):
            clipped_grads = [(tf.clip_by_norm(gv[0], max_grad_norm), gv[1]) for gv in grads]
        return clipped_grads


def average_gradients(tower_grads, skip_none=False):
    """
      From tensorflow cifar 10 tutorial codes
      Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
      skip_none: Boolean. Whether to throw away non gradient variable or raise exception.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Skip None gradients
        if grad_and_vars[0][0] is None and skip_none:
            colorlog.warning("%s has None gradient" % grad_and_vars[0][1].name)
            continue

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def load_pretrained_word_vector(cfg):

    if os.path.isfile(cfg.cached_emb_fname):
        colorlog.info("Load pretrained word vector ({})".format(cfg.cached_emb_fname))
        pretrained_word_vector = load_cached_vector(cfg.cached_emb_fname)
    else:
        colorlog.info("Cache and Load pretrained word vector ({})".format(cfg.cached_emb_fname))
        with open('data/audiocaps/features/auxiliary/' + str(cfg.vocab_size) + '.vocab', 'r') as vocab_fp:
            vocab = set(map(str.strip, vocab_fp.readlines()))

        embed_dict = {}
        num_exists = 0
        with open('data/pretrained_word_vectors/fastText.commoncrawl/crawl-300d-2M.vec', 'r') as in_fp:
            lines = in_fp.readlines()
            for line in lines:
                line = line.strip().split()

                key = line[0]
                value = np.array(list(map(float, line[1:])), dtype=np.float32)

                if key in vocab:
                    embed_dict[key] = value
                    num_exists += 1

        # colorlog.info("Cache {} / {} embeddings".format(num_exists, len(vocab)))

        with open(cfg.cached_emb_fname, 'wb') as out_fp:
            pickle.dump(embed_dict, out_fp)
            pretrained_word_vector = None

        colorlog.info("Cache Succes")
        pretrained_word_vector = load_cached_vector(cfg.cached_emb_fname)

    return pretrained_word_vector

