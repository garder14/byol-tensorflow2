import tensorflow as tf



def byol_loss(p, z):
    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)

    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    return 2 - 2 * tf.reduce_mean(similarities)
