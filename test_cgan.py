### -*- coding: utf-8 -*-
"""
test_cgan.py

Testing a Conditional GAN model.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function
import artefacts
import data_processing
from config import ConfigCGAN as config
import cgan as model
import utils
import tensorflow as tf
#tf.enable_eager_execution()
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import math
import sys
from load_data import load, load_images, load_labels


def generator_d_loss(generated_output):
    # [1,1,...,1] with generated images since we want the discriminator to judge them as real
    return tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


def generator_abs_loss(labels, generated_images):
    # As well as "fooling" the discriminator, we want particular pressure on ground-truth accuracy
    return config.L1_lambda * tf.compat.v1.losses.absolute_difference(labels, generated_images)  # mean


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since we want our generated examples to look like it
    real_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

#def compute_psnr(x1, x2, max_diff=1):
#    return 20. * tf.math.log(max_diff / data_processing.rmse(x1, x2)) / tf.math.log(10.)

def train_step(inputs, labels):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(inputs, training=True)

        real_output = discriminator(labels, training=True)
        generated_output = discriminator(generated_images, training=True)
            
        gen_d_loss = generator_d_loss(generated_output)
        gen_abs_loss = generator_abs_loss(labels, generated_images)
        gen_loss = gen_d_loss + gen_abs_loss
        gen_rmse = data_processing.rmse(labels, generated_images)
        gen_psnr = data_processing.psnr(labels, generated_images)
        disc_loss = discriminator_loss(real_output, generated_output)

        # Logging
        global_step.assign_add(1)
        log_metric(gen_d_loss, "train/loss/generator_deception")
        log_metric(gen_abs_loss, "train/loss/generator_abs_error")
        log_metric(gen_loss, "train/loss/generator")
        log_metric(disc_loss, "train/loss/discriminator")
        log_metric(gen_rmse, "train/accuracy/rmse")
        log_metric(gen_psnr, "train/accuracy/psnr")

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        #batched_dataset = dataset.batch(4)
        for x, y in dataset:
            train_step(x, y)

        generate_and_save_images(generator,
                                 epoch + 1,
                                 selected_inputs,
                                 selected_labels)
        
        # saving (checkpoint) the model every few epochs
        if (epoch + 1) % config.save_per_epoch == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        print ('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                          time.time()-start))
    # generating after the final epoch
    generate_and_save_images(generator,
                             epochs,
                             selected_inputs,
                             selected_labels)


def generate_and_save_images(model, epoch, test_inputs, test_labels):
    print("Saving images")
    if model is None:
        predictions = test_inputs
    else:
        # Make sure the training parameter is set to False because we
        # don't want to train the batchnorm layer when doing inference.
        predictions = model(test_inputs, training=False)

    types = [predictions, test_labels]  # Image types (alternated in rows)
    ntype = len(types)
    nrows = 4
    ncols = 8
    fig = plt.figure(figsize=(8, 5))
    
    for i in range(ntype * predictions.shape[0]):
        plt.subplot(nrows, ncols, i+1)
        # Get relative index
        row = int(i / ncols)
        row_rel = row % ntype
        group = int(row / ntype)
        shift = ncols * (group * (ntype - 1) + row_rel)
        idx = i - shift
        # Plot
        for t in range(ntype):
            if row_rel == 0:
                j = int(i / ntype)
                rmse = data_processing.rmse(test_labels[j], predictions[j], norm=2)
                psnr = data_processing.psnr(test_labels[j], predictions[j], max_diff=1)
                plt.xlabel('RMSE={:.3f}\nPSNR={:.2f}'.format(rmse, psnr), fontsize=8)
            if row_rel == t:
                plt.imshow(types[row_rel][idx, :, :, 0], vmin=-1, vmax=1, cmap='gray')
                break
        plt.xticks([])
        plt.yticks([])
    
    plt.savefig(os.path.join(results_path, 'image_at_epoch_{:04d}.png'.format(epoch)))
    # plt.show()


def log_metric(value, name):
    with summary_writer.as_default():
        tf.summary.scalar(name, value, step=global_step)

def filter_by_label(images, labels, label, new_label):
    images = images[np.where(labels == label)]
    return images, np.full(images.shape[0], new_label)

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def prepare_data(data_dir):
    #(X_train, Y_train), (X_test, Y_test) = mnist.load_data()    
    #https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
    (X_train, Y_train), (X_test, Y_test) = load(
      os.path.join(data_dir, "emnist-digits-train-images-idx3-ubyte.gz"),
      os.path.join(data_dir, "emnist-digits-train-labels-idx1-ubyte.gz"),
      os.path.join(data_dir, "emnist-digits-test-images-idx3-ubyte.gz"),
      os.path.join(data_dir, "emnist-digits-test-labels-idx1-ubyte.gz")
    )

    (X_train_letters, Y_train_letters), (X_test_letters, Y_test_letters) = load(
      os.path.join(data_dir, "emnist-letters-train-images-idx3-ubyte.gz"),
      os.path.join(data_dir, "emnist-letters-train-labels-idx1-ubyte.gz"),
      os.path.join(data_dir, "emnist-letters-test-images-idx3-ubyte.gz"),
      os.path.join(data_dir, "emnist-letters-test-labels-idx1-ubyte.gz")
    )

    print(len(X_train),len(Y_train),len(X_test),len(Y_test))
    print(len(X_train_letters),len(Y_train_letters),len(X_test_letters),len(Y_test_letters))
    print(Y_train_letters[:100])
    X_train_letters, Y_train_letters = filter_by_label(X_train_letters, Y_train_letters, 24, 10 ) # 'x' = 24
    X_test_letters, Y_test_letters = filter_by_label(X_test_letters, Y_test_letters, 24, 10)
    print(len(X_train_letters),len(Y_train_letters),len(X_test_letters),len(Y_test_letters))
    print(Y_train_letters[:100])
    
    X_train = np.concatenate((X_train, X_train_letters))
    Y_train = np.concatenate((Y_train, Y_train_letters))
    X_test = np.concatenate((X_test, X_test_letters))
    Y_test = np.concatenate((Y_test, Y_test_letters))

    shuffle_in_unison(X_train, Y_train)
    shuffle_in_unison(X_test, Y_test)

    return (X_train, Y_train), (X_test, Y_test)

if __name__ == '__main__':        
    # model_path = "out/noise_gan/model/2018-12-12-11-07-49"
    parser = argparse.ArgumentParser(description='emnist experiment')
    parser.add_argument('--epochs', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--data_dir', type=str, default='$DATA_DIR',
                        help='Directory with data')
    parser.add_argument('--result_dir', type=str, default='$RESULT_DIR',
                        help='Directory with results')

    args = parser.parse_args()
    if (args.result_dir[0] == '$'):
        RESULT_DIR = os.environ[args.result_dir[1:]]
    else:
        RESULT_DIR = args.result_dir

    model_path = os.path.join(RESULT_DIR, 'model')
    if (args.data_dir[0] == '$'):
        DATA_DIR = os.environ[args.data_dir[1:]]
    else:
        DATA_DIR = args.data_dir


    # Make directories for this run
    time_string = "cgan"#time.strftime("%Y-%m-%d-%H-%M-%S")
    #model_path = os.path.join(config.model_path, time_string)
    model_path = os.path.join(RESULT_DIR, time_string, 'model')
    results_path = os.path.join(RESULT_DIR, time_string, 'results')
    utils.safe_makedirs(model_path)
    utils.safe_makedirs(results_path)

    # Initialise logging
    log_path = os.path.join(RESULT_DIR, time_string, 'logs')
    summary_writer = tf.summary.create_file_writer(log_path, flush_millis=10000) #tf.contrib.summary.create_file_writer(log_path, flush_millis=10000)
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Load the dataset
    #(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    (train_images, _), (test_images, _) = prepare_data(DATA_DIR)

    # Add noise for condition input
    #train_inputs = artefacts.add_gaussian_noise(train_images, stdev=0.2, data_range=(0, 255)).astype('float32')
    train_inputs = artefacts.add_noise(train_images).reshape(train_images.shape[0], 
                                        config.raw_size,
                                        config.raw_size,
                                        config.channels).astype("float32")
    train_inputs = data_processing.normalise(train_inputs, (-1, 1), (0, 255))
    #train_inputs = train_inputs.astype('float32')
    train_images = train_images.reshape(train_images.shape[0], 
                                        config.raw_size,
                                        config.raw_size,
                                        config.channels)
    train_images = data_processing.normalise(train_images.astype('float32'), (-1, 1), (0, 255))
    train_labels = train_images
    #train_images = data_processing.normalise(train_images, (-1, 1), (0, 255))
    #train_labels = train_images.astype('float32')

    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))\
        .shuffle(args.batch_size).batch(args.batch_size)

    # Test set
    #test_inputs = artefacts.add_gaussian_noise(test_images, stdev=0.2, data_range=(0, 255)).astype('float32')
    test_inputs = artefacts.add_noise(test_images).reshape(test_images.shape[0], 
                                        config.raw_size,
                                        config.raw_size,
                                        config.channels).astype("float32")
    test_inputs = data_processing.normalise(test_inputs, (-1, 1), (0, 255))
    #test_inputs = test_inputs.astype("float32")
    test_images = test_images.reshape(test_images.shape[0], 
                                    config.raw_size,
                                    config.raw_size,
                                    config.channels)
    #test_images = data_processing.normalise(test_images, (-1, 1), (0, 255))
    #test_labels = test_images.astype('float32')
    test_labels = data_processing.normalise(test_images.astype('float32'), (-1, 1), (0, 255))
    # Set up some random (but consistent) test cases to monitor
    num_examples_to_generate = 16
    random_indices = np.random.choice(np.arange(test_inputs.shape[0]),
                                    num_examples_to_generate,
                                    replace=False)
    selected_inputs = test_inputs[random_indices]
    selected_labels = test_labels[random_indices]

    # Set up the models for training
    generator = model.make_generator_model_small()
    discriminator = model.make_discriminator_model()

    generator_optimizer = tf.optimizers.Adam(config.learning_rate)
    discriminator_optimizer = tf.optimizers.Adam(config.learning_rate)

    checkpoint_prefix = os.path.join(model_path, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    #print(train_images[0])
    generate_and_save_images(None, 0, selected_inputs, selected_labels)  # baseline
    print("\nTraining...\n")
    # Compile training function into a callable TensorFlow graph (speeds up execution)
    #train_step = tf.contrib.eager.defun(train_step)
    train(train_dataset, args.epochs)
    print("\nTraining done\n")
    
    model.save(os.path.join(model_path, 'model.h5'), save_format='h5')
    os.system(f'(cd {model_path};tar cvfz ../saved_model.tgz .)')
    print(str(os.listdir(os.environ['RESULT_DIR'])))
    print(os.environ['RESULT_DIR'])
    sys.stdout.flush()

    # checkpoint.restore(tf.train.latest_checkpoint(model_path))
    # prediction = generator(selected_inputs, training=False)

    # for i in range(num_examples_to_generate):
    #     fig = plt.figure()
    #     plt.subplot(1, 4, 1)
    #     plt.imshow(selected_inputs[i, :, :, 0], vmin=-1, vmax=1)
    #     plt.subplot(1, 4, 2)
    #     plt.imshow(prediction[i, :, :, 0], vmin=-1, vmax=1)
    #     plt.subplot(1, 4, 3)
    #     plt.imshow(selected_labels[i, :, :, 0], vmin=-1, vmax=1)
    #     plt.subplot(1, 4, 4)
    #     plt.imshow(abs(selected_labels[i, :, :, 0] - prediction[i, :, :, 0]), vmin=0, vmax=2)
    #     plt.show()

    # generate_and_save_images(generator, 0, selected_inputs, selected_labels)
