import tensorflow as tf



def random_crop_flip_resize(image):
    # Random cropping
    h_crop = tf.cast(tf.random.uniform(shape=[], minval=13, maxval=33, dtype=tf.int32), tf.float32)
    w_crop = h_crop * tf.random.uniform(shape=[], minval=0.67, maxval=1.0)
    h_crop, w_crop = tf.cast(h_crop, tf.int32), tf.cast(w_crop, tf.int32)
    opposite_aspectratio = tf.random.uniform(shape=[])
    if opposite_aspectratio < 0.5:
        h_crop, w_crop = w_crop, h_crop
    image = tf.image.random_crop(image, size=[h_crop, w_crop, 3])

    # Horizontal flipping
    horizontal_flip = tf.random.uniform(shape=[])
    if horizontal_flip < 0.5:
        image = tf.image.random_flip_left_right(image)

    # Resizing to original size
    image = tf.image.resize(image, size=[32, 32])
    return image


def random_color_distortion(image):
    # Random color jittering (strength 0.5)
    color_jitter = tf.random.uniform(shape=[])
    if color_jitter < 0.8:
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.clip_by_value(image, 0, 1)

    # Color dropping
    color_drop = tf.random.uniform(shape=[])
    if color_drop < 0.2:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.tile(image, [1, 1, 3])

    return image


@tf.function
def augment_image_pretraining(image):
    image = random_crop_flip_resize(image)
    image = random_color_distortion(image)
    return image


@tf.function
def augment_image_finetuning(image):
    image = random_crop_flip_resize(image)
    return image