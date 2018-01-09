from django.core.files.base import File
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.shortcuts import render, redirect
from django.http import HttpResponse
import datetime
import os

from stronka.models import Baza

def index(request):
    datetime_now = datetime.datetime.now()
    context = {
        'datetime_now' : datetime_now,
    }
    return render(request, 'index.html', context)

def upload(request):
    if request.method == 'POST':

        success = True
        error_msg = ''

        image = request.FILES.get('uploaded_file')

        if type(image) is not InMemoryUploadedFile:
            error_msg = 'Nie można wczytać obrazka obrazka.'
            success = False

        # Baza.resetNumerObrazka()

        image_nr = Baza.nextNumerObrazka()

        fname, fext = os.path.splitext(image.name)

        if fext != '.jpg':
            error_msg = 'Obrazek musi mieć rozszerzenie .jpg.'
            success = False

        filename1 = 'image_{0:05d}_0_BEFORE{1}'.format(image_nr, fext)
        filename2 = 'image_{0:05d}_1_AFTER{1}'.format(image_nr, fext)
        filepath1 = os.path.join('media', filename1)
        filepath2 = os.path.join('media', filename2)

        with open(filepath1, 'wb') as f:
            f.write(image.read())

        object_detect(filepath1, filepath2)

        context = {
            'success': success,
            'error_msg': error_msg,
            'image1': filepath1,
            'image2': filepath2,
        }

        return render(request, 'upload.html', context)

    return redirect('/')

def object_detect(file1, file2):
    import numpy as np
    import tensorflow as tf
    import scipy.misc
    from PIL import Image

    if tf.__version__ != '1.4.0':
        raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util

    MODEL_NAME = 'training'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = MODEL_NAME + '/object-detection.pbtxt'
    NUM_CLASSES = 1

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name = '')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes = NUM_CLASSES, use_display_name = True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    with detection_graph.as_default():
        with tf.Session(graph = detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            image = Image.open(file1)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis = 0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict = {image_tensor: image_np_expanded}
            )

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates = True,
                line_thickness = 4
            )

            scipy.misc.imsave(file2, image_np)
