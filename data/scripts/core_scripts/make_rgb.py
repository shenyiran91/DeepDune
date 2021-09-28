import numpy as np
import os
from PIL import Image
from tempfile import TemporaryFile


data_dir = "/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles"

img_dir = data_dir+'/filtered_image_tiles_test/'
label_dir = data_dir+'/filtered_label_tiles_test/'

image_list = []
label_list = []

#Finds and sort all files in datafolder subfolders (image_tiles and label_tiles)                                                                                                                                                    
img_names = sorted(os.listdir(img_dir))
img_names.sort(key = lambda x: x.split('_',)[7])

label_names = sorted(os.listdir(label_dir))
label_names.sort(key = lambda x: x.split('_',)[4])

for ind,img in enumerate(img_names):

        #Opens images as Image objects.                                                                                                                                                                                             
        image = Image.open(img_dir+img).convert("L")
        label = Image.open(label_dir+label_names[ind]).convert("L")
                
        image.save("/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles/filtered_image_tiles_test_gs/"+str(img))
        label.save("/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles/filtered_label_tiles_test_gs/"+str(label_names[ind]))

        print("Image and label for ", str(img), " is done.")
