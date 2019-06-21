import tensorflow as tf

from models.base import BaseNetwork
from helpers.nlp_helper import sequence_reconstruction_error
from helpers.enc_dec_helper import rnn_decoding, single_rnn_cell, attention_decoder_cell
from helpers.layer_helper import attention_flow_layer


class PyramidLSTM(BaseNetwork):
    def __init__(self,
                 cfg,
                 index_to_string,
                 iterators,
                 iters_in_epoch,
                 is_training):
        self.alpha_list = []
        super(PyramidLSTM, self).__init__(cfg, index_to_string, iterators, iters_in_epoch, is_training)

    def _build_towers(self):
        self._build_towers_audiocaps()

    def _build(self, iterator):
        with tf.name_scope("raw_inputs"):
            answer = iterator.answer  # GO_ID + seq + EOS_ID
            answer_length = iterator.answer_length  # Include GO + EOS length
            feature_FC3 = iterator.feature_FC3
            feature_C4 = iterator.feature_C4 # [?, ?, 8, 12, 512]
            FC3_length = iterator.FC3_length
            C4_length = iterator.C4_length
            access_id = iterator.access_id
            labels = iterator.labels  # [num_labels]
            num_labels = iterator.num_labels  # scalar


        answer_max_length = tf.reduce_max(answer_length)
        audio_length = tf.constant(10, tf.int32, [self.cfg.batch_size])

        layer1_input_dim = feature_C4.shape.as_list()
        with tf.variable_scope("encoder_0"):
            feature_in = feature_FC3
            feature_len = FC3_length

            feature_in_shape = feature_in.shape.as_list()

            # Bi-LSTM encoder
            fw_cell = single_rnn_cell(self.cfg.encoder_unit_type, self.cfg.L0_encoder_num_units,
                                    self.cfg.encoder_dropout, "train" if self.is_training else "test")
            bw_cell = single_rnn_cell(self.cfg.encoder_unit_type, self.cfg.L0_encoder_num_units,
                                    self.cfg.encoder_dropout, "train" if self.is_training else "test")
            l0_encoder_outputs, l0_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, feature_in, feature_len, dtype=tf.float32)

            # concat the fw, bw outputs.
            l0_encoder_outputs = tf.concat(l0_encoder_outputs, axis=2)

            l0_squashed_state_0 = tf.layers.dense(
                tf.concat([l0_encoder_state[0].h, l0_encoder_state[1].h], axis=1),
                layer1_input_dim[-1], name="l0_h_projection_0")
            l0_squashed_state_0 = tf.expand_dims(l0_squashed_state_0, axis=1)

        with tf.variable_scope("encoder_1"):
            feature_in = feature_C4
            feature_len = C4_length
            feature_in = tf.reshape(feature_in, [-1,
                feature_in.shape.as_list()[1], feature_in.shape.as_list()[2] *
                feature_in.shape.as_list()[3], feature_in.shape.as_list()[4]])
            feature_in = tf.reduce_sum(feature_in, axis=2)


            feature_in = feature_in + l0_squashed_state_0

            # Bi-LSTM encoder
            fw_cell = single_rnn_cell(self.cfg.encoder_unit_type, self.cfg.L1_encoder_num_units,
                                    self.cfg.encoder_dropout, "train" if self.is_training else "test")
            bw_cell = single_rnn_cell(self.cfg.encoder_unit_type, self.cfg.L1_encoder_num_units,
                                    self.cfg.encoder_dropout, "train" if self.is_training else "test")
            l1_encoder_outputs, l1_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, feature_in, feature_len, dtype=tf.float32)
            l1_encoder_outputs = tf.concat(l1_encoder_outputs, axis=2)


            encoder_outputs = l1_encoder_outputs
            encoder_state = l1_encoder_state

        with tf.variable_scope("word_encoder"):
            labels_embedding = tf.nn.embedding_lookup(self.emb_word, labels)
            # Bi-LSTM encoder
            word_fw_cell = single_rnn_cell(self.cfg.encoder_unit_type, self.cfg.word_encoder_num_units,
                                        self.cfg.word_encoder_dropout, "train" if self.is_training else "test")
            word_bw_cell = single_rnn_cell(self.cfg.encoder_unit_type, self.cfg.word_encoder_num_units,
                                        self.cfg.word_encoder_dropout, "train" if self.is_training else "test")
            word_encoder_outputs, word_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                word_fw_cell, word_bw_cell, labels_embedding, num_labels, dtype=tf.float32)
            word_encoder_outputs = tf.concat(word_encoder_outputs, axis=2)
        word_encoder_tuple = (word_encoder_outputs, word_encoder_state, num_labels)



        # Attention flow
        audio2word, word2audio = attention_flow_layer(encoder_outputs, audio_length,
                                                    word_encoder_outputs, num_labels)
        mixed_encoder_outputs = tf.concat([encoder_outputs, audio2word,
                                        encoder_outputs * audio2word,
                                        encoder_outputs * word2audio], axis=2)


        with tf.variable_scope("context_encoder"):
            context_fw_cell = single_rnn_cell(self.cfg.encoder_unit_type, self.cfg.context_encoder_num_units,
                                            self.cfg.context_encoder_dropout, "train" if self.is_training else "test")
            context_bw_cell = single_rnn_cell(self.cfg.encoder_unit_type, self.cfg.context_encoder_num_units,
                                            self.cfg.context_encoder_dropout, "train" if self.is_training else "test")
            context_encoder_outputs, context_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                context_fw_cell, context_bw_cell, mixed_encoder_outputs, audio_length, dtype=tf.float32)
            context_encoder_outputs = tf.concat(context_encoder_outputs, axis=2)
            context_encoder_outputs = tf.concat([context_encoder_outputs, mixed_encoder_outputs], axis=2)
        context_encoder_tuple = (context_encoder_outputs, context_encoder_state, audio_length)

        with tf.variable_scope("decoder_cell"):
            decoder_rnn_cell = single_rnn_cell(self.cfg.decoder_unit_type, self.cfg.decoder_num_units,
                                            self.cfg.decoder_dropout, "train" if self.is_training else "test")

        # Output attention
        encoder_tuples = [word_encoder_tuple, context_encoder_tuple]

        decoder_initial_state = None

        # Decoding
        train_decoder_cell, train_decoder_initial_state = self._build_decoder_cell(
            encoder_tuples, "train", decoder_initial_state=decoder_initial_state, reuse=False)
        output_lengths, logits, sample_id = rnn_decoding(
            answer, answer_length, train_decoder_cell,
            train_decoder_initial_state, self.emb_word,
            self.cfg.max_length, mode='train', reuse=False)
        reconstruction_error = sequence_reconstruction_error(logits, answer[:, 1:], answer_length - 1,
                                                             average=True, smoothing_rate=self.cfg.label_smoothing,
                                                             vocab_size=self.cfg.vocab_size)


        if self.is_training:
            if "train_0" in tf.contrib.framework.get_name_scope():
                family = "train_loss"
                tf.summary.scalar("reconstruction_error", reconstruction_error, family=family)
                tf.summary.histogram("argmax", sample_id, family=family)
            return reconstruction_error
        else:
            test_decoder_cell, test_decoder_initial_state = self._build_decoder_cell(
                encoder_tuples, "beam_search", decoder_initial_state=decoder_initial_state,  reuse=True)

            # For prediction/GT visualization
            test_output_lengths, test_logits, test_sample_id = rnn_decoding(
                answer, answer_length, test_decoder_cell,
                test_decoder_initial_state, self.emb_word,
                self.cfg.max_length, mode='beam_search', beam_width=self.cfg.beam_width, reuse=True)
            test_output_max_length = tf.reduce_max(test_output_lengths)

            prediction_words = self.pad_word_outputs(test_sample_id, test_output_max_length)
            answer_words = self.pad_word_outputs(answer[:, 1:], answer_max_length)

            return answer_words, prediction_words, reconstruction_error, access_id


    def _build_decoder_cell(self, encoder_tuples,
                            mode, decoder_initial_state, reuse):
        beam_width = self.cfg.beam_width if mode == "beam_search" else 1
        batch_size = self.cfg.batch_size * beam_width
        with tf.variable_scope('decoder_cell', reuse=reuse):
            rnn_cell, decoder_initial_state = attention_decoder_cell(
                encoder_tuples, self.cfg.decoder_attention_type,
                self.cfg.decoder_unit_type, self.cfg.decoder_num_units, self.cfg.decoder_dropout,
                "train" if self.is_training else "test", self.cfg.batch_size, beam_width,
                decoder_initial_state=decoder_initial_state, reuse=reuse)

        return rnn_cell, decoder_initial_state



