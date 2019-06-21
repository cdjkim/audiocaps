import tensorflow as tf
import pickle


def load_cached_vector(fname):
    with open(fname, 'rb') as fp:
        word_dict = pickle.load(fp)
    return word_dict


def load_fasttext(fname):
    word_dict = {}
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip().split()

            key = line[0]
            value = map(float, line[1:])

            word_dict[key] = value

    return word_dict


def sequence_reconstruction_error(decoder_logits, answer, answer_length, average=True, smoothing_rate=0.0, vocab_size=15000):
    onehot_answer = tf.one_hot(answer, vocab_size)

    if smoothing_rate > 0:
        smooth_positives = 1.0 - smoothing_rate
        smooth_negatives = smoothing_rate / vocab_size
        onehot_answer = onehot_answer * smooth_positives + smooth_negatives

    xentropy = tf.losses.softmax_cross_entropy(onehot_answer, decoder_logits, label_smoothing=0,
                                               reduction=tf.losses.Reduction.NONE)
    answer_mask = tf.sequence_mask(answer_length, dtype=decoder_logits.dtype)
    if average:
        reconstruction_error = tf.reduce_sum(xentropy * answer_mask) / tf.to_float(tf.reduce_sum(answer_mask))
    else:
        reconstruction_error = tf.reduce_sum(xentropy * answer_mask, axis=1)
        reconstruction_error = tf.reduce_mean(reconstruction_error)
    return reconstruction_error
