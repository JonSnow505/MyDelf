# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Extracts DELF features from a list of images, saving them to file.

The images must be in JPG format. The program checks if descriptors already
exist, and skips computation for those.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from google.protobuf import text_format
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.python.platform import app
import time

from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io
from delf import feature_pb2

from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import plot_matches

from flask import Flask, request, jsonify
import base64
app = Flask(__name__)

# Extension of feature files.
_DELF_EXT = '.delf'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100

_DISTANCE_THRESHOLD = 0.7

##def _ReadImageList(list_path):
##  """Helper function to read image paths.
##
##  Args:
##    list_path: Path to list of images, one image path per line.
##
##  Returns:
##    image_paths: List of image paths.
##  """
##  with tf.gfile.GFile(list_path, 'r') as f:
##    image_paths = f.readlines()
##  image_paths = [entry.rstrip() for entry in image_paths]
##  return image_paths

tf.logging.set_verbosity(tf.logging.INFO)

# Read list of images.
##tf.logging.info('Reading list of images...')
##image_paths = _ReadImageList('list_images_func.txt')
##num_images = len(image_paths)
##tf.logging.info('done! Found %d images', num_images)

# Parse DelfConfig proto.
config = delf_config_pb2.DelfConfig()
with tf.gfile.FastGFile('delf_config_example.pbtxt', 'r') as f:
  text_format.Merge(f.read(), config)

# Create output directory if necessary.
if not os.path.exists('data/'):
  os.makedirs('data/')

# Tell TensorFlow that the model will be built into the default Graph.
with tf.Graph().as_default():
  # Reading list of images.
##  filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
##  reader = tf.WholeFileReader()

  with tf.Session() as sess:
    # Initialize variables.
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Loading model that will be used.
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                               config.model_path)
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name('input_image:0')
    input_score_threshold = graph.get_tensor_by_name('input_abs_thres:0')
    input_image_scales = graph.get_tensor_by_name('input_scales:0')
    input_max_feature_num = graph.get_tensor_by_name(
        'input_max_feature_num:0')
    boxes = graph.get_tensor_by_name('boxes:0')
    raw_descriptors = graph.get_tensor_by_name('features:0')
    feature_scales = graph.get_tensor_by_name('scales:0')
    attention_with_extra_dim = graph.get_tensor_by_name('scores:0')
    attention = tf.reshape(attention_with_extra_dim,
                           [tf.shape(attention_with_extra_dim)[0]])

    locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
        boxes, raw_descriptors, config)

    locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(
        'data/target_phy.delf')

    locations_11, _, descriptors_11, _, _ = feature_io.ReadFromFile(
        'data/target_geo.delf')

    num_features_1 = locations_1.shape[0]
    tf.logging.info("Loaded image 1's %d features" % num_features_1)

    num_features_11 = locations_11.shape[0]
    tf.logging.info("Loaded image 11's %d features" % num_features_11)

    d1_tree = cKDTree(descriptors_1)
    d11_tree = cKDTree(descriptors_11)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    @app.route("/")
    def hello():
      return "Hello World!"

    # We'll support /resnet
    @app.route("/resnet", methods=['POST'])
    def myfunc():
      
      value = request.form['image']
      data = base64.b64decode(value)
      image_tf = tf.image.decode_image(data, channels=3)
      im = sess.run(image_tf)
##
##      elapsed = (time.clock() - start)
##      tf.logging.info('Processing image, took %f seconds', elapsed)

##      image_tf = tf.image.decode_image(value, channels=3)
##      im = sess.run(image_tf)
      # Extract and save features.
      (locations_2, descriptors_2, feature_scales_out,
       attention_out) = sess.run(
           [locations, descriptors, feature_scales, attention],
           feed_dict={
               input_image:
                   im,
               input_score_threshold:
                   config.delf_local_config.score_threshold,
               input_image_scales:
                   list(config.image_scales),
               input_max_feature_num:
                   config.delf_local_config.max_feature_num
           })

##      out_desc_filename = 'image' + _DELF_EXT
##      out_desc_fullpath = os.path.join('data/', out_desc_filename)
##      
##      serialized_desc = feature_io.WriteToFile(
##          out_desc_fullpath, locations_out, feature_scales_out,
##          descriptors_out, attention_out)

      # Read features.
      num_features_2 = locations_2.shape[0]
      tf.logging.info("Loaded image 2's %d features" % num_features_2)

      # Find nearest-neighbor matches using a KD tree.
      distances, indices = d1_tree.query(
          descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

      # Select feature locations for putative matches.
      locations_2_to_use = np.array([
          locations_2[i,] for i in range(num_features_2)
          if indices[i] != num_features_1
      ])
      locations_1_to_use = np.array([
          locations_1[indices[i],] for i in range(num_features_2)
          if indices[i] != num_features_1
      ])

      # Perform geometric verification using RANSAC.
      model_robust, inliers1 = ransac(
          (locations_1_to_use, locations_2_to_use),
          AffineTransform,
          min_samples=3,
          residual_threshold=20,
          max_trials=1000)

      tf.logging.info('Found %d inliers for phy' % sum(inliers1))

      # Find nearest-neighbor matches using a KD tree.
      distances, indices = d11_tree.query(
          descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

      # Select feature locations for putative matches.
      locations_2_to_use = np.array([
          locations_2[i,] for i in range(num_features_2)
          if indices[i] != num_features_11
      ])
      locations_11_to_use = np.array([
          locations_11[indices[i],] for i in range(num_features_2)
          if indices[i] != num_features_11
      ])

      # Perform geometric verification using RANSAC.
      model_robust, inliers11 = ransac(
          (locations_11_to_use, locations_2_to_use),
          AffineTransform,
          min_samples=3,
          residual_threshold=20,
          max_trials=1000)

      tf.logging.info('Found %d inliers for geo' % sum(inliers11))

      # Visualize correspondences, and save to file.
##      fig, ax = plt.subplots()
##      img_1 = mpimg.imread('target.jpg')
##      img_2 = mpimg.imread('image.jpg')
##      inlier_idxs = np.nonzero(inliers)[0]
##      plot_matches(
##          ax,
##          img_1,
##          img_2,
##          locations_1_to_use,
##          locations_2_to_use,
##          np.column_stack((inlier_idxs, inlier_idxs)),
##          matches_color='b')
##      ax.axis('off')
##      ax.set_title('DELF correspondences')
##      plt.savefig('match.jpg')

      if sum(inliers1) >= sum(inliers11):
        return 'phy;' + str(sum(inliers1))
      else:
        return 'geo;' + str(sum(inliers11))

    app.run(host='0.0.0.0', port=5000)

    # Finalize enqueue threads.
    coord.request_stop()
    coord.join(threads)
