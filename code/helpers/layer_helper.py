import tensorflow as tf


def attention_flow_layer(source, source_length, target, target_length, name='attention_flow_layer', reuse=False):
    """
    Attention-flow layer from following paper
    Bi-Directional Attention Flow for Machine Comprehension, ICLR 2017
    https://arxiv.org/abs/1611.01603

    Args:
        source: [batch, source_length, embed_dim]
        source_length: [batch]
        target: [batch, target_length, embed_dim]
        target_length: [batch]
    """

    source_max_length = tf.reduce_max(source_length)
    target_max_length = tf.reduce_max(target_length)
    embed_dim = source.shape.as_list()[2]
    with tf.variable_scope(name):
        tiled_source = tf.transpose(tf.tile(tf.expand_dims(source, -1), [1, 1, 1, target_max_length]),
                                    perm=[0, 1, 3, 2])  # [batch, source_len, target_len, emb]
        tiled_target = tf.transpose(tf.tile(tf.expand_dims(target, -1), [1, 1, 1, source_max_length]),
                                    perm=[0, 3, 1, 2])  # [batch, source_len, target_len, emb]

        tiled_mixed = tf.concat([tiled_source, tiled_target, tiled_source * tiled_target], axis=3)  # [batch, source_len, target_len, emb*3]
        mixed_logits = tf.layers.dense(tiled_mixed, 1, activation=None, use_bias=False,
                                       name="score_function")  # [batch, source_len, target_len, 1]

        # source2target
        tiled_source2target_att = tf.tile(tf.nn.softmax(mixed_logits, axis=2), [1, 1, 1, embed_dim])  # [batch, source_len, target_len, embed]
        source2target = tf.reduce_max(tiled_source2target_att * tiled_target, axis=2)  # [batch, source_len, embed]

        # target2source
        tiled_target2source_att = tf.tile(tf.nn.softmax(tf.reduce_max(mixed_logits, axis=2), axis=1),
                                          [1, 1, embed_dim])  # [batch, source_len, embed]
        target2source = tf.tile(tf.reduce_sum(tiled_target2source_att * source, axis=1, keepdims=True),
                                [1, source_max_length, 1])  # [batch, source_len, embed]

    return source2target, target2source



def layer_norm_compute_python(x, epsilon, scale, bias):
    """
    Layer norm raw computation.
    (https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py)
    """
    epsilon, scale, bias = [tf.cast(t, x.dtype) for t in [epsilon, scale, bias]]
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
    """
    Layer normalize the tensor x, averaging over the last dimension.
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py
    """
    if filters is None:
        filters = x.get_shape().as_list()[-1]
    with tf.variable_scope(name, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable("layer_norm_scale", [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable("layer_norm_bias", [filters], initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result


