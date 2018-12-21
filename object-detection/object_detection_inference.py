# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:

import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import label_map_util

tf.app.flags.DEFINE_string('test_data_path', '../../../train-images', 'Images directory')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './output_inference_graph', '')
tf.app.flags.DEFINE_string('output_dir', './od-output', '')

FLAGS = tf.app.flags.FLAGS


# # Model preparation

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:



# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# In[9]:
def get_images():
    PATH_TO_TEST_IMAGES_DIR = FLAGS.test_data_path
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    TEST_IMAGE_PATHS = []
    for root, dirs, files in os.walk(PATH_TO_TEST_IMAGES_DIR):
        for file in files:
            for ext in exts:
                if file.endswith(ext):
                    TEST_IMAGE_PATHS.append(os.path.join(root, file))
                    break
    print('Find {} images'.format(len(TEST_IMAGE_PATHS)))
    return TEST_IMAGE_PATHS


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
    MODEL_NAME = FLAGS.checkpoint_path
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    print(FLAGS.test_data_path, FLAGS.output_dir)
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'oid_bbox_trainable_label_map.pbtxt')
    NUM_CLASSES = 1000
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # ## Load a (frozen) Tensorflow model into memory.
    # In[6]:
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    classes_detected = dict()
    # In[26]:
    with detection_graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            TEST_IMAGE_PATHS = get_images()
            for image_path in TEST_IMAGE_PATHS:
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.

                start = time.time()
                image = cv2.imread(image_path)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})
                # print('[Net timing] {}'.format(time.time()-start))

                # start = time.time()
                # all outputs are float32 numpy arrays, so convert types as appropriate
                classes = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                boxes = output_dict['detection_boxes'][0]
                scores = output_dict['detection_scores'][0]

                output_list = []
                for i in range(boxes.shape[0]):
                    if scores is None or scores[i] > 0.1:
                        if classes[i] in category_index.keys():
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        display_str = str(class_name)
                        # ymin, xmin, ymax, xmax = box
                        # box_coord = (left, right, top, bottom)
                        try:
                            classes_detected[display_str] += 1
                        except KeyError:
                            classes_detected[display_str] = 1
                        output_list.append(
                            str(classes[i]) + ',' + "{0:.2f}".format(scores[i]) + ',' + display_str + ':' + ','.join(
                                map(str, boxes[i])))
                if len(output_list) > 0:
                    res_file = os.path.join(
                        FLAGS.output_dir,
                        '{}.txt'.format(
                            os.path.basename(image_path).split('.')[0]))
                    with open(res_file, 'w') as f:
                        f.write('\n'.join(output_list))

                # print('[Writing timing] {}'.format(time.time() - start))

    res_file = os.path.join(
        FLAGS.output_dir,'{}.txt'.format('meta-classes'))
    with open(res_file, 'w') as f:
        for key, value in sorted(classes_detected.items()):
            f.write(str(key) + ',' + str(value) + '\n')


if __name__ == '__main__':
    tf.app.run()
