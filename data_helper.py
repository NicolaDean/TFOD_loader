
import matplotlib
import matplotlib.pyplot as plt

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf

from models.research.object_detection.utils import visualization_utils as viz_utils


def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.8)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)


'''
    rf = Roboflow(api_key="D5jpG7thd1uxwm3apfHd")
    project = rf.workspace("jacob-solawetz").project("aerial-maritime")
    dataset = project.version(9).download("tfrecord")

    load_dataset("D5jpG7thd1uxwm3apfHd","jacob-solawetz","aerial-maritime")
'''
def load_dataset(api_key,workspace_id,project_id):
    #Downloading data from Roboflow
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_id).project(project_id)
    dataset = project.version(9).download("tfrecord")

    return rf,dataset, project

def get_data_path(dataset,file_name="movable-objects"):
    if dataset == None:
       return "","",""
    # NOTE: Update these TFRecord names from "cells" and "cells_label_map" to your files!
    test_record_fname       = dataset.location + f'/test/{file_name}.tfrecord'
    train_record_fname      = dataset.location + f'/train/{file_name}.tfrecord'
    label_map_pbtxt_fname   = dataset.location + f'/train/{file_name}_label_map.pbtxt'

    return test_record_fname, train_record_fname, label_map_pbtxt_fname