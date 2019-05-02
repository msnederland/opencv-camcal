import cv2
import numpy as np
import os
import tensorflow as tf
import sys
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class ObjectDetector:
  @classmethod
  def label(self, contours, comment=None):
    approx = cv2.approxPolyDP(contours, 0.01*cv2.arcLength(contours, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    if len(approx) == 3:
        shape =  "Triangle"
    elif len(approx) == 4:
        shape =  "Rectangle"
    elif len(approx) == 5:
        shape =  "Pentagon"
    elif len(approx) == 6:
        shape =  "Cube"
    elif 7 < len(approx) < 15:
        shape =  "Ellipse"
    else:
        shape =  "Circle"

    return shape + comment if comment else shape

  def draw_bounding_box(self, image, contours):
    x,y,w,h = cv2.boundingRect(contours)
    comment = " (%s, %s)" % (x,y)

    cv2.rectangle(image, (x,y), (x+w,y+w), (0,200,0), 1)
    cv2.putText(image, self.label(contours, comment), (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
    return image

  def detect(self, image):
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img_grey, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
      self.draw_bounding_box(image, contour)

    return image


class tfDectector:
  @classmethod
  def detect(self, image):
    CWD_PATH = os.getcwd()

    PATH_TO_CKPT = os.path.join(CWD_PATH,'model', 'frozen_inference_graph.pb')

    PATH_TO_LABELS = os.path.join(CWD_PATH,'model','labelmap.pbtxt')

    NUM_CLASSES = 3

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    frame_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    return_image  = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)
    return image