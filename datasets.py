import tensorflow as tf

from augmentation import augment_image_pretraining, augment_image_finetuning



class CIFAR10:

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.num_train_images, self.num_test_images = self.y_train.shape[0], self.y_test.shape[0]
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'sheep', 'truck']

        # Normalize training and testing images
        self.x_train = tf.cast(self.x_train / 255., tf.float32)
        self.x_test = tf.cast(self.x_test / 255., tf.float32)

        self.y_train = tf.cast(tf.squeeze(self.y_train), tf.int32)
        self.y_test = tf.cast(tf.squeeze(self.y_test), tf.int32)


    def get_batch_pretraining(self, batch_id, batch_size):
        augmented_images_1, augmented_images_2 = [], []
        for image_id in range(batch_id*batch_size, (batch_id+1)*batch_size):
            image = self.x_train[image_id]
            augmented_images_1.append(augment_image_pretraining(image))
            augmented_images_2.append(augment_image_pretraining(image))
        x_batch_1 = tf.stack(augmented_images_1)
        x_batch_2 = tf.stack(augmented_images_2)
        return x_batch_1, x_batch_2  # (bs, 32, 32, 3), (bs, 32, 32, 3)


    def get_batch_finetuning(self, batch_id, batch_size):
        augmented_images = []
        for image_id in range(batch_id*batch_size, (batch_id+1)*batch_size):
            image = self.x_train[image_id]
            augmented_images.append(augment_image_finetuning(image))
        x_batch = tf.stack(augmented_images)
        y_batch = tf.slice(self.y_train, [batch_id*batch_size], [batch_size])
        return x_batch, y_batch  # (bs, 32, 32, 3), (bs)


    def get_batch_testing(self, batch_id, batch_size):
        x_batch = tf.slice(self.x_test, [batch_id*batch_size, 0, 0, 0], [batch_size, -1, -1, -1])
        y_batch = tf.slice(self.y_test, [batch_id*batch_size], [batch_size])
        return x_batch, y_batch  # (bs, 32, 32, 3), (bs)


    def shuffle_training_data(self):
        random_ids = tf.random.shuffle(tf.range(self.num_train_images))
        self.x_train = tf.gather(self.x_train, random_ids)
        self.y_train = tf.gather(self.y_train, random_ids)
