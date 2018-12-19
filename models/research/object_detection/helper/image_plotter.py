
import os

import cv2
import numpy as np

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt
import sys
import tensorflow as tf

from tqdm import tnrange, tqdm_notebook
from pprint import pprint

def image_test_and_plot(image_manager, sess, NUM_BYTES_FOR_MASK):

    dim_ = 100
    n = 4
    canvas_orig = np.empty((dim_ * n, dim_ * n, NUM_BYTES_FOR_MASK))
    canvas_recon = np.empty((dim_ * n, dim_ * n, NUM_BYTES_FOR_MASK))
    for i in range(n):
        batch_x = image_manager.next_batch(n)
        batch_x = batch_x[:, :, :, :NUM_BYTES_FOR_MASK]
        batch_x = batch_x.reshape(batch_x.shape[0], -1).astype(np.float16)

        output = tf.get_default_graph().get_tensor_by_name("decoder:0")
        ip_img= tf.get_default_graph().get_tensor_by_name("input:0")
        g = output.eval(feed_dict={ip_img: batch_x})

        for j in range(n):
            canvas_orig[i * dim_:(i + 1) * dim_, j * dim_:
                        (j + 1) * dim_,:] = batch_x[j].reshape([dim_, dim_, NUM_BYTES_FOR_MASK])

        for j in range(n):
            canvas_recon[i * dim_:(i + 1) * dim_, j * dim_:
                         (j + 1) * dim_,:] = g[j].reshape([dim_, dim_, NUM_BYTES_FOR_MASK])
    print(np.linalg.norm(canvas_orig - canvas_recon)/n)

    
    print("Original Images vs Reconstructed Images")  
    for i in range(NUM_BYTES_FOR_MASK):
        print('Layer:',i)
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(canvas_orig[:,:,i], origin="upper", cmap="gray")
        axarr[1].imshow(canvas_recon[:,:,i], origin="upper", cmap="gray")
        plt.show()
    
