import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layers_core

from utils import vocab_utils
from helpers.layer_helper import layer_norm


def single_rnn_cell(unit_type, num_units, dropout, mode, residual_connection=False, reuse=False):
    """Create an instance of a single RNN cell.

    Args:
        - dropout: 0.0 means no dropout
        - mode: "train", "val", "test". This is used for dropout
    """
    dropout = dropout if mode == "train" else 0.0

    # Cell type
    if unit_type == "lstm":
        single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, reuse=reuse)
    elif unit_type == "gru":
        single_cell = tf.contrib.rnn.GRUCell(num_units, reuse=reuse)
    elif unit_type == "layer_norm_lstm":
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, layer_norm=True, reuse=reuse)
    else:
        raise ValueError("Unkown unit type {}!".format(unit_type))

    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))

    # Residual
    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)

    return single_cell


def attention_decoder_cell_fn(decoder_rnn_cell, memories, attention_type,
                              decoder_type, decoder_num_units, decoder_dropout,
                              mode, batch_size, beam_width=1, decoder_initial_state=None, reuse=False):
    """Create an decoder cell with attention. It takes decoder cell as argument

    Args:
        - memories: (encoder_outputs, encoder_state, input_length) tuple
        - attention_type: "luong", "bahdanau"
        - mode: "train", "test"
    """
    if mode == "train":
        beam_width = 1
    with tf.variable_scope('attention_decoder_cell', reuse=reuse):
        attention_mechanisms = []
        attention_layers = []
        for idx, (encoder_outputs, encoder_state, input_length) in enumerate(memories):
            # Tile batch for beam search, if beam_width == 1, then nothing happens
            encoder_outputs, input_length, encoder_state, beam_batch_size = prepare_beam_search_decoder_inputs(
                beam_width, encoder_outputs, input_length, encoder_state, batch_size)

            # Temporal attention along time step
            if attention_type == "luong":
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    decoder_num_units, memory=encoder_outputs, memory_sequence_length=input_length)
            elif attention_type == "bahdanau":
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    decoder_num_units, memory=encoder_outputs, memory_sequence_length=input_length)
            attention_layer = tf.layers.Dense(decoder_num_units, name="{}th_attention".format(idx),
                                              use_bias=False, dtype=tf.float32, _reuse=reuse)
            attention_mechanisms.append(attention_mechanism)
            attention_layers.append(attention_layer)

        #decoder_rnn_cell = single_rnn_cell(decoder_type, decoder_num_units, decoder_dropout, mode, reuse=reuse)
        attention_rnn_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_rnn_cell, attention_mechanisms, attention_layer=attention_layers,
            initial_cell_state=None, name="AttentionWrapper")

        # Set decoder initial state
        initial_state = attention_rnn_cell.zero_state(dtype=tf.float32, batch_size=beam_batch_size)
        if decoder_initial_state:
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(decoder_initial_state, multiplier=beam_width)
            initial_state = initial_state.clone(cell_state=decoder_initial_state)

    return attention_rnn_cell, initial_state


def attention_decoder_cell(memories, attention_type,
                           decoder_type, decoder_num_units, decoder_dropout,
                           mode, batch_size, beam_width=1, decoder_initial_state=None, reuse=False):
    """Create an default decoder cell with attention

    Args:
        - memories: (encoder_outputs, encoder_state, input_length) tuple
        - attention_type: "luong", "bahdanau"
        - mode: "train", "test"
    """
    with tf.variable_scope('decoder_cell', reuse=reuse):
        decoder_rnn_cell = single_rnn_cell(decoder_type, decoder_num_units, decoder_dropout, mode, reuse=reuse)
    return attention_decoder_cell_fn(decoder_rnn_cell, memories, attention_type,
                                     decoder_type, decoder_num_units, decoder_dropout,
                                     mode, batch_size, beam_width, decoder_initial_state, reuse)


def prepare_beam_search_decoder_inputs(beam_width, memory, source_sequence_length, encoder_state, batch_size):
    """
    Tile encoder outputs for beam search decoder. If beam_width == 1, then nothing happens
    https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/AttentionWrapper

    """
    memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
    source_sequence_length = tf.contrib.seq2seq.tile_batch(source_sequence_length, multiplier=beam_width)
    encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
    batch_size = batch_size * beam_width
    return memory, source_sequence_length, encoder_state, batch_size


def rnn_decoding(answer, answer_length, cell, initial_state, emb_word,
                 max_length, mode, beam_width=1, reuse=False):
    """

    Args:
        answer: includes GO and EOS
        answer_length: includes EOS
    """
    batch_size = answer.get_shape()[0]
    vocab_size = emb_word.get_shape()[0]

    with tf.variable_scope("rnn_decoder"):
        output_layer = tf.layers.Dense(vocab_size, use_bias=False,
                                       name="output_projection", _reuse=reuse)
        maximum_iterations = None if mode == "train" else max_length

        if mode in ["train", "greedy"]:
            if mode == "train":
                answer_embeddings = tf.nn.embedding_lookup(emb_word, answer)
                helper = tf.contrib.seq2seq.TrainingHelper(answer_embeddings[:, :-1], answer_length - 1)
            elif mode == "greedy":
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    emb_word,
                    start_tokens=tf.tile([vocab_utils.GO_ID], [batch_size]),
                    end_token=vocab_utils.EOS_ID)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell, helper=helper,
                                                      initial_state=initial_state,
                                                      output_layer=output_layer)
        elif mode == "beam_search":
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=cell,
                embedding=emb_word,
                start_tokens=tf.tile([vocab_utils.GO_ID], [batch_size]),
                end_token=vocab_utils.EOS_ID,
                initial_state=initial_state,
                beam_width=beam_width,
                output_layer=output_layer)
        else:
            raise NotImplementedError()
        outputs, _, output_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                       output_time_major=False,
                                                                       maximum_iterations=maximum_iterations)

    if mode == "beam_search":
        logits = tf.no_op()
        sample_id = outputs.predicted_ids[:, :, 0]  # NOTE: currently, only use best result
    else:
        logits = outputs.rnn_output
        sample_id = outputs.sample_id

    return output_lengths, logits, sample_id

