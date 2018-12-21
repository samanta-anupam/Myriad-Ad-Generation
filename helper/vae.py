from helper.image_plotter import image_test_and_plot
import tensorflow as tf
import os
from helper.config import *
import cv2
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import matplotlib.pyplot as plt
import sys

from tqdm import tnrange, tqdm_notebook
from helper.image_manager import ImageManager
from pprint import pprint
from ipywidgets import *
from IPython.display import display
from IPython.html import widgets
# %matplotlib inline
import math

learning_rate = 1e-4
num_steps = 20
batch_size = 16
# Network Parameters
NUM_BYTES_FOR_MASK = 3+1
image_dim = 100*100*NUM_BYTES_FOR_MASK # images are 100x100 pixels
hidden_dim1 = 1024
hidden_dim2 = 128
latent_dim = 40
beta = 0.3

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Variables
weights = {
    'encoder_h1': tf.Variable(glorot_init([image_dim, hidden_dim1])),
    'encoder_h2': tf.Variable(glorot_init([hidden_dim1, hidden_dim2])),
    'z_mean': tf.Variable(glorot_init([hidden_dim2, latent_dim])),
    'z_std': tf.Variable(glorot_init([hidden_dim2, latent_dim])),
    'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim2])),
    'decoder_h2': tf.Variable(glorot_init([hidden_dim2, hidden_dim1])),
    'decoder_out': tf.Variable(glorot_init([hidden_dim1, image_dim]))
}
biases = {
    'encoder_b1': tf.Variable(glorot_init([hidden_dim1])),
    'encoder_b2': tf.Variable(glorot_init([hidden_dim2])),
    'z_mean': tf.Variable(glorot_init([latent_dim])),
    'z_std': tf.Variable(glorot_init([latent_dim])),
    'decoder_b1': tf.Variable(glorot_init([hidden_dim2])),
    'decoder_b2': tf.Variable(glorot_init([hidden_dim1])),
    'decoder_out': tf.Variable(glorot_init([image_dim]))
}

# Define VAE Loss
def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
#     x_reconstructed = tf.clip_by_value(x_reconstructed, 1e-7, 1 - 1e-7)
    encode_decode_loss = x_true * tf.log(1e-7 + x_reconstructed) \
                         + (1 - x_true) * tf.log(1e-7 + 1 - x_reconstructed)

    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -beta * 0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)


MODEL_PATH = os.path.join(OUT_TF_DATA,'models')+'/'+'_'.join(['vae',str(hidden_dim1),str(hidden_dim2),str(latent_dim),'sigmoid','adam'])

# Building the encoder
input_image = tf.placeholder(tf.float32, shape=[None, image_dim], name='input')
encoder1 = tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1']
encoder1 = tf.nn.leaky_relu(encoder1)
# encoder1 = tf.nn.dropout(encoder1, 0.5)
encoder2 = tf.matmul(encoder1, weights['encoder_h2']) + biases['encoder_b2']
encoder = tf.nn.leaky_relu(encoder2, name='encoder')
# encoder2 = tf.nn.dropout(encoder, 0.5)
z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

# Sampler: Normal (gaussian) random distribution
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
z = tf.add(z_mean, tf.exp(z_std / 2) * eps, name='hidden')


# Building the decoder (with scope to re-use these layers later)
decoder1 = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']
decoder1 = tf.nn.leaky_relu(decoder1)
# decoder1 = tf.nn.dropout(decoder1, 0.5)
decoder2 = tf.matmul(decoder1, weights['decoder_h2']) + biases['decoder_b2']
decoder2 = tf.nn.leaky_relu(decoder2)
# decoder2 = tf.nn.dropout(decoder2, 0.5)
decoder = tf.matmul(decoder2, weights['decoder_out']) + biases['decoder_out']
decoder = tf.nn.sigmoid(decoder, name='decoder')

# plt.show()
# plt.tight_layout()

def train(image_manager, test_image_manager):
    n_samples = image_manager.files.shape[0]
    batch_loss_plot = []
    test_loss_plot = []
    
    test_images = test_image_manager.next_batch(16)[:, :, :, :NUM_BYTES_FOR_MASK]
    test_images = test_images.reshape(test_images.shape[0], -1).astype(np.float16)

    loss_op = vae_loss(decoder, input_image)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()
    
    canvas_recon = np.zeros((100 * 4, 100 * 4, NUM_BYTES_FOR_MASK))
    for j in range(16):
        # Draw the generated digits 
        x = j//4
        y = j%4
    #         print(test_images[j].reshape([100, 100, NUM_BYTES_FOR_MASK]).shape)
    #         print(test_images[j].shape)
        canvas_recon[y * 100:(y + 1) * 100, x * 100:
                     (x + 1) * 100,:] = test_images[j].reshape([100, 100, NUM_BYTES_FOR_MASK])
    f, axarr = plt.subplots(2,2)

    for i in range(NUM_BYTES_FOR_MASK):
        axarr[i//2][i%2].imshow(canvas_recon[:,:,i], origin="upper", cmap="gray")

    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        saver = tf.train.Saver()

        fig,ax = plt.subplots(3,3,figsize=(8,8))
    #     plt.ion()

        for k in range(NUM_BYTES_FOR_MASK):
            ax[(k+5)//3][(k+5)%3].clear()
            ax[(k+5)//3][(k+5)%3].imshow(canvas_recon[:,:,k], origin="upper", cmap="gray")

    #     fig.show()
        fig.canvas.draw()
        # Training
        try:
            for i in tqdm_notebook(range(num_steps)):
                # Prepare Data
                # Get the next batch of MNIST data (only images are needed, not labels)
                avg_loss = 0.
                total_batch = n_samples // batch_size
        #         total_batch = 2000

                # Loop over all batches
                for j in range(total_batch):
                    batch_x = image_manager.next_batch(batch_size)
                    batch_x = batch_x[:, :, :, :NUM_BYTES_FOR_MASK]
                    batch_x = batch_x.reshape(batch_x.shape[0], -1).astype(np.float16)
                    # Train
                    feed_dict = {input_image: batch_x}
                    _, l = sess.run([train_op, loss_op], feed_dict=feed_dict)
                    avg_loss += l / n_samples * batch_size
                    batch_loss_plot.append(l)
                    if (j%(total_batch//10))==0:
                        ax[0][0].clear()
                        ax[0][0].plot(batch_loss_plot[-total_batch:])
                        g = decoder.eval(feed_dict={input_image: batch_x})
                        for k in range(16):
                            # Draw the generated digits 
                            x = k//4
                            y = k%4
                            canvas_recon[y * 100:(y + 1) * 100, x * 100:
                                         (x + 1) * 100,:] = g[k].reshape([100, 100, NUM_BYTES_FOR_MASK])
                        for k in range(4):
                            ax[(k+1)//3][(k+1)%3].clear()
                            ax[(k+1)//3][(k+1)%3].imshow(canvas_recon[:,:,k], origin="upper", cmap="gray")
                        fig.canvas.draw()

                validation_error = loss_op.eval(feed_dict={input_image: test_images})

                g = decoder.eval(feed_dict={input_image: test_images})
                for j in range(16):
                    # Draw the generated digits 
                    x = j//4
                    y = j%4
                    canvas_recon[y * 100:(y + 1) * 100, x * 100:
                                 (x + 1) * 100,:] = g[j].reshape([100, 100, NUM_BYTES_FOR_MASK])
                for k in range(NUM_BYTES_FOR_MASK):
                    ax[(k+5)//3][(k+5)%3].clear()
                    ax[(k+5)//3][(k+5)%3].imshow(canvas_recon[:,:,k], origin="upper", cmap="gray")
                    fig.canvas.draw()

                test_loss_plot.append(validation_error)
        except KeyboardInterrupt:
            print('Stopped')
        save_ckpt(sess, saver)
        
        image_test_and_plot(image_manager, sess, NUM_BYTES_FOR_MASK)
        print(test_loss_plot)

        p = plt.figure()
        plt.plot(test_loss_plot, label='validation')
        plt.plot(batch_loss_plot, label='minibatch-acc')
        plt.legend()
        plt.show()

def save_ckpt(sess, saver):
    model_name = MODEL_PATH +'/model.ckpt'
    save_path = saver.save(sess, model_name)

    export_dir = MODEL_PATH
    inputs = {'input': input_image}
    hidden = {'hidden': encoder}
    outputs = {'output': decoder}
    print('Saved model at:', save_path)


def restore_and_test(test_image_manager):
    # MODEL_PATH = '/gpfs/scratch/asamanta/models/vae_1024_128_32_sigmoid_adam'
    tf.reset_default_graph()
    with tf.Session() as sess:
      # Initialize v1 since the saver will not.
    #   v1.initializer.run()
        saver = tf.train.import_meta_graph(MODEL_PATH+'/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
        image_test_and_plot(test_image_manager, sess, NUM_BYTES_FOR_MASK)
        
def get_model_tensors():
    tf.reset_default_graph()
    sess = tf.Session()
      # Initialize v1 since the saver will not.
    #   v1.initializer.run()
    saver = tf.train.import_meta_graph(MODEL_PATH+'/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
    return tf.get_default_graph().get_tensor_by_name("decoder:0"), tf.get_default_graph().get_tensor_by_name("input:0"), tf.get_default_graph().get_tensor_by_name("hidden:0"), sess


