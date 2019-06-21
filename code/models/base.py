import tensorflow as tf
import colorlog

from helpers.etc_helper import clip_grads, average_gradients


class BaseNetwork(object):
    def __init__(self,
                 cfg,
                 index_to_string,
                 iterators,
                 iters_in_epoch,
                 is_training):


        self.cfg = cfg
        self.index_to_string = index_to_string
        self.iterators = iterators
        self.iters_in_epoch = iters_in_epoch
        self.is_training = is_training

        # Global params
        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.maximum(
            self.cfg.init_lr * tf.pow(cfg.lr_decay_factor, tf.floordiv(self.global_step, iters_in_epoch * self.cfg.num_epochs_per_decay)),
            self.cfg.min_lr)  # step decay

        self.opt = tf.train.AdamOptimizer(self.lr)

        # Initializer
        initializer = tf.glorot_normal_initializer()
        tf.get_variable_scope().set_initializer(initializer)

        # Embeddings
        with tf.device("/cpu:0"):
            self._build_embeddings()

        self.train_summaries = []
        self._build_towers()

        # Update summary
        if self.is_training:
            self.train_summaries.append(tf.summary.scalar("lr", self.lr, family="etc"))

        # Saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.cfg.max_to_keep)

        # Print trainable variables
        if self.is_training:
            params = tf.trainable_variables()
            for param in params:
                print("  %s, %s, %s" % (param.name, str(param.get_shape()), param.op.device))

    def train(self, sess):
        _, summary, global_step, lr = sess.run([self.train_op, self.train_summary, self.global_step, self.lr])

        return {'summaries': [summary], 'global_step': global_step, 'lr': lr}

    def test(self, sess):
        return sess.run({'prediction': self.sentence_inference_op,
                        'answer': self.sentence_answer_op,
                        'perplexity': self.test_reconstruction_error_op,
                        'access_id': self.access_id_op})

    # Overrided
    def _build_towers(self):
        raise NotImplementedError()

    # Overrided
    def _build(self):
        raise NotImplementedError()

    def _build_towers_audiocaps(self):
        """
        In-graph data parallelization for audio2text task
        """
        mode = 'train' if self.is_training else 'test'
        tower_grads = []
        tower_answer_words = []
        tower_generated_words = []
        tower_reconstruction_error = []
        tower_crossentropy_error = []
        tower_access_id = []
        tower_accuracy = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.cfg.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (mode, i)) as scope:
                        colorlog.info("Build %s_%d" % (mode, i))
                        if self.is_training:
                            total_loss = self._build(self.iterators[i])
                            grads = self.opt.compute_gradients(total_loss)
                            tower_grads.append(grads)

                            if i == 0:
                                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        else:
                            answer_words, generated_words, reconstruction_error, access_id = self._build(self.iterators[i])
                            tower_answer_words.append(answer_words)
                            tower_generated_words.append(generated_words)
                            tower_reconstruction_error.append(reconstruction_error)
                            tower_access_id.append(access_id)

                        # Reuse variables for the next tower
                        tf.get_variable_scope().reuse_variables()

        if self.is_training:
            averaged_grads = average_gradients(tower_grads, True)
            clipped_grads = clip_grads(averaged_grads, self.cfg.max_grad_norm)

            self.train_op = self.opt.apply_gradients(clipped_grads, global_step=self.global_step)
            self.train_summaries += summaries
            self.train_summary = tf.summary.merge(self.train_summaries)
        else:
            self.sentence_answer_op = tf.concat(tower_answer_words, 0)  # [num_gpus*batch_size, legnth]
            self.sentence_inference_op = tf.concat(tower_generated_words, 0)  # [num_gpus*batch_size, length]
            self.test_reconstruction_error_op = tf.reduce_mean(tower_reconstruction_error)
            self.access_id_op = tf.concat(tower_access_id, 0)  # [num_gpus*batch_size]

    def _build_embeddings(self):
        with tf.variable_scope('embedding_matrix'):
            self.emb_word = tf.get_variable(name='emb_word', shape=[self.cfg.vocab_size, self.cfg.word_embed_size],
                                            dtype=tf.float32, trainable=True)

    def apply_word_vector(self, graph, sess, pretrained_word_vector):
        applied_word_vector = {}
        with graph.as_default():
            word_embedding = sess.run(self.emb_word)
            indices_op = tf.range(self.cfg.vocab_size, dtype=tf.int64)
            words_op = self.index_to_string.lookup(indices_op)
            words = sess.run(words_op)

            valid_counter = 0
            for idx, word in enumerate(words):
                word = word.decode()
                if word in pretrained_word_vector:
                    pretrained_vector = pretrained_word_vector[word]
                    word_embedding[idx] = pretrained_vector
                    valid_counter += 1
                    applied_word_vector[word] = pretrained_vector
            assign_word_vector = tf.assign(self.emb_word, word_embedding)
            sess.run(assign_word_vector)

        colorlog.info("Use %d word embeddings from pretrained word vector" % valid_counter)
        return applied_word_vector

    def pad_word_outputs(self, outputs, output_max_length):
        padding = tf.zeros([self.cfg.batch_size, self.cfg.max_length - output_max_length + 1], dtype=tf.int64)
        padded_words = self.index_to_string.lookup(tf.concat([tf.to_int64(outputs), padding], axis=1))
        return padded_words
