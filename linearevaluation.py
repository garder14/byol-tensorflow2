import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import tensorflow as tf

from datasets import CIFAR10
from models import ResNet18, ResNet34, ClassificationHead



encoders = {'resnet18': ResNet18, 'resnet34': ResNet34}


def compute_test_accuracy(data, f_net, c_net):
    batch_size = 500
    num_batches = data.num_test_images // batch_size

    num_correct_predictions = 0
    for batch_id in range(num_batches):
        x, y = data.get_batch_testing(batch_id, batch_size)
        h = f_net(x, training=False)
        y_pred_logits = c_net(h)
        y_pred_labels = tf.argmax(y_pred_logits, axis=1, output_type=tf.int32)

        num_correct_predictions += tf.reduce_sum(tf.cast(tf.equal(y_pred_labels, y), tf.int32))

    return tf.cast(num_correct_predictions / data.num_test_images, tf.float32)


def main(args):

    # Load CIFAR-10 dataset
    data = CIFAR10()

    # Define hyperparameters
    num_epochs = 50
    batch_size = 512

    # Instantiate networks f and c
    f_net = encoders[args.encoder]()
    c_net = ClassificationHead()

    # Initialize the weights of f and c
    x, y = data.get_batch_finetuning(batch_id=0, batch_size=batch_size)
    h = f_net(x, training=False)
    print('Shape of h:', h.shape)
    s = c_net(h)
    print('Shape of s:', s.shape)

    # Load the weights of f from pretraining
    f_net.load_weights(args.encoder_weights)
    print('Weights of f loaded.')


    # Define optimizer
    batches_per_epoch = data.num_train_images // batch_size
    total_update_steps = num_epochs * batches_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(5e-2, total_update_steps, 5e-4, power=2)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    
    @tf.function
    def train_step_evaluation(x, y):  # (bs, 32, 32, 3), (bs)

        # Forward pass
        with tf.GradientTape() as tape:
            h = f_net(x, training=False)  # (bs, 512)
            y_pred_logits = c_net(h)  # (bs, 10)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred_logits))
        
        # Backward pass
        grads = tape.gradient(loss, c_net.trainable_variables)
        opt.apply_gradients(zip(grads, c_net.trainable_variables))

        return loss


    log_every = 10  # batches
    for epoch_id in range(num_epochs):
        data.shuffle_training_data()
        
        for batch_id in range(batches_per_epoch):
            x, y = data.get_batch_finetuning(batch_id, batch_size)
            loss = train_step_evaluation(x, y)
            if (batch_id + 1) % log_every == 0:
                print('[Epoch {}/{} Batch {}/{}] Loss: {:.4f}'.format(epoch_id+1, num_epochs, batch_id+1, batches_per_epoch, loss))
    
    # Compute classification accuracy on test set
    test_accuracy = compute_test_accuracy(data, f_net, c_net)
    print('Test Accuracy: {:.4f}'.format(test_accuracy))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder', type=str, required=True, choices=['resnet18', 'resnet34'], help='Encoder architecture')
    parser.add_argument('--encoder_weights', type=str, help='Encoder weights')

    args = parser.parse_args()
    main(args)
