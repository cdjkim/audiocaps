import tensorflow as tf

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def create_vocab_tables_with_input(vocab_file, vocab_size):
    vocab_placeholder = tf.placeholder(tf.int64, [None, None])

    #string_to_index_table = tf.contrib.lookup.index_table_from_file(
    #    vocab_file, vocab_size=vocab_size, default_value=UNK_ID)
    index_to_string_table = tf.contrib.lookup.index_to_string_table_from_file(
        vocab_file, vocab_size=vocab_size, default_value=_UNK)

    index_to_string = index_to_string_table.lookup(vocab_placeholder)

    return index_to_string, vocab_placeholder


def create_vocab_tables(vocab_file, vocab_size):
    string_to_index_table = tf.contrib.lookup.index_table_from_file(
        vocab_file, vocab_size=vocab_size, default_value=UNK_ID)
    index_to_string_table = tf.contrib.lookup.index_to_string_table_from_file(
        vocab_file, vocab_size=vocab_size, default_value=_UNK)

    return string_to_index_table, index_to_string_table
