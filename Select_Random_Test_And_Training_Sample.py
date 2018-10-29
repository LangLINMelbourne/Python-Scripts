#Author: Lang LIN
#Usage: 1.python Select_Random_Test_And_Training_Sample
#       2.python Select_Random_Test_And_Training_Sample --image_directory="image folder" ----train_directory="training set folder" --test_directory = "testing set folder"
#Effect: automatically select 10% of image and copy them into testing set folder and rest 90% will be copy to training set folder
#        default image folder is "./image/",default training folder is "./image/train/", default testing folder is "./image/test/"
#Careful: this code should only run once otherwise data will be incorrupt. Also can delete everything in train & test folder then rerun the program to make new dataset

from pdf2image import convert_from_path
import glob
import argparse
import os
import random
import shutil

parser = argparse.ArgumentParser(description='Enter directory for input pdf files & directory for output image files')
parser.add_argument('-id','--image_directory',default='./images/',help='image directory, default is ./images/')
parser.add_argument('-tr','--train_directory',default='./images/train/',help='directory for coping 90 percent image as training set,default is ./images/train/')
parser.add_argument('-te','--test_directory',default='./images/test/',help='directory for coping 10 percent image as testing set,default is ./images/test/')

args = vars(parser.parse_args())

image_path = os.path.dirname(args['image_directory'])
train_path = os.path.dirname(args['train_directory'])
test_path = os.path.dirname(args['test_directory'])

try:
    os.stat(train_path)
except:
    os.mkdir(train_path)
try:
    os.stat(test_path)
except:
    os.mkdir(test_path)

jpg_files = glob.glob(image_path+"/*.jpg")
xml_files = glob.glob(image_path+"/*.xml")

#currently only contains jpg files but there might be other kinds of image file types like .png will be added into database.
files = sorted(jpg_files)
file_amount = len(files)
test_amount = file_amount/10
train_amount = file_amount-test_amount

random_test_list_without_repetition = sorted(random.sample(range(file_amount),test_amount))
random_test_file_names = [files[i] for i in random_test_list_without_repetition]

#train set contain all element in total set but not in test set, also pop from largest index can prevent pop wrong element
random_train_file_names = files
for test_case_image_number in reversed(random_test_list_without_repetition):
    random_train_file_names.pop(test_case_image_number)

for img_name in random_test_file_names:
    xml_name = img_name.replace(".jpg",".xml")
    if xml_name in xml_files:
        new_img_name = img_name.replace(args['image_directory'],args['test_directory'])
        new_xml_name = xml_name.replace(args['image_directory'],args['test_directory'])
        shutil.copy(img_name,new_img_name)
        shutil.copy(xml_name,new_xml_name)

for img_name in random_train_file_names:
    xml_name = img_name.replace(".jpg",".xml")
    if xml_name in xml_files:
        new_img_name = img_name.replace(args['image_directory'],args['train_directory'])
        new_xml_name = xml_name.replace(args['image_directory'],args['train_directory'])
        shutil.copy(img_name,new_img_name)
        shutil.copy(xml_name,new_xml_name)
