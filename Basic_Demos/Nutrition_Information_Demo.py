import sys
import zipfile
import tarfile
import pytesseract
import argparse
import cv2
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from utils import label_map_util
from utils import visualization_utils as vis_util
import six.moves.urllib as urllib
from distutils.version import StrictVersion
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        select_image_button = QPushButton('Select Image File', self)
        select_image_button.resize(300,150)
        select_image_button.move(10, 10)

        recognize_location_button = QPushButton('Recognize Selected Image \nNutrition Information Table Location', self)
        recognize_location_button.resize(300,150)
        recognize_location_button.move(10, 260)

        cut_image_button = QPushButton('Cut Original Image \nAnd Only Show One Cutted Image', self)
        cut_image_button.resize(300,150)
        cut_image_button.move(10, 510)

        text_recognize_button = QPushButton('Recognize Text In Cutted Image', self)
        text_recognize_button.resize(300,150)
        text_recognize_button.move(10, 760)




        original_image_label = QLabel(self)
        original_image = QPixmap('sample_original_image.jpg')
        resized_original_image = original_image.scaled(350, 350)
        original_image_label.setPixmap(resized_original_image)
        original_image_label.move(350,25)

        detected_image_label = QLabel(self)
        detected_image = QPixmap('sample_detected_image.jpg')
        resized_detected_image = detected_image.scaled(350, 350)
        detected_image_label.setPixmap(resized_detected_image)
        detected_image_label.move(800,25)

        cutted_image_label = QLabel(self)
        cutted_image = QPixmap('sample_cutted_image.jpg')
        resized_cutted_image = cutted_image.scaled(350, 350)
        cutted_image_label.setPixmap(resized_cutted_image)
        cutted_image_label.move(350,520)

        recognized_text_label = QLabel(self)
        recognized_text_label.setText("test")
        recognized_text_label.setAlignment(Qt.AlignCenter)
        recognized_text_label.resize(350,350)
        recognized_text_label.move(800,520)
        #sample_text = self.rotationRecognize("sample_cutted_image.jpg")
        #sample_text = self.rebuildText(sample_text)
        sample_text = '. % DAILY\nQUANTITY INTAKE* QUANTITY\nPERSERVING (PER SERVING) PER 100g\nENERGY 412 kJ 47% 2020 kJ\nPROTEIN 1.29 24% 9.8 g\nFAT, TOTAL 3.99 5.5 % 18.9 g\n-SATURATED 2.2 g 9.1% 10.7 g\nCARBOHYDRATE 143 g 4.6% 69.9 g\n- SUGARS 12g 8.0 % 35.1 g\nSODIUM 36 mg 1.6 % 177 mg\nE ADULT DIET OF 8700K).'
        recognized_text_label.setText(sample_text)


        self.resize(1200, 920)
        self.center()

        self.setWindowTitle('Nutrition_Information_Demo')
        self.show()





    def patternRecognize():
        #part 1 "Variables"
        # What model to download.
        MODEL_NAME = 'nutrition_information'
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('data', 'object_detection.pbtxt')
        NUM_CLASSES = 1

        #part 2 "Load a (frozen) Tensorflow model into memory"
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        #part 3 "Loading label map"
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        #part 4 "Helper Code" shows below because it is a def function

        #part 5 "Detection"
        #part 5.1
        PATH_TO_TEST_IMAGES_DIR = 'test_images'
        TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 17) ]
        # Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)
        #part 5.2 shows below because it is a def function
        #part 5.3
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            width, height = image.size
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = self.load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = self.run_inference_for_single_image(image_np, detection_graph)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            cv2.imwrite(image_path+"detected.jpg", image_np)

            area = (int(width*output_dict['detection_boxes'][0][1]),
            int(height*output_dict['detection_boxes'][0][0]),
            int(width*output_dict['detection_boxes'][0][3]),
            int(height*output_dict['detection_boxes'][0][2]))
            cutted_img = image.crop(area)
            cutted_img.show()
            cutted_image_np = self.load_image_into_numpy_array(cutted_img)
            cv2.imwrite(image_path+"cutted.jpg", cutted_image_np)
            cutted_img.save("test_images/cutted.jpg")

    #part 4 "Helper Code"
    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    #part 5.2
    def run_inference_for_single_image(self,image, graph):
        with graph.as_default():
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
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


#this function rotate the image 20 degree to left and 20 degree to right and for each degree it will use pytesseract try
#to recognize the text and reason to do that is to prevent user take photo with deviation in angle(pytesseract cannot regonize well with angle)
    def rotationRecognize(self,image_path):
        img = Image.open(image_path);
        text = ""
        for i in range(20):
            temp_text = pytesseract.image_to_string(img.rotate(i))
            if(len(text)<len(temp_text)):
                text=temp_text
            temp_text = pytesseract.image_to_string(img.rotate(360-i))
            if(len(text)<len(temp_text)):
                text=temp_text
        return text

#this function remove all extra \n and space in recognized text to make it more readable
    def rebuildText(self,text):
        splited_text = text.split("\n")
        for i in range(len(splited_text)):
            splited_text[i] = splited_text[i].strip()
        new_text = ""
        for i in range(len(splited_text)):
            if(splited_text[i] != ""):
                new_text = new_text + splited_text[i] + "\n"
        return new_text

    def center(self):

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
