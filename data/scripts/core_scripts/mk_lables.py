import numpy as np
from PIL import Image
import os

main_dir = '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles'
img_dir = main_dir+'/filtered_image_chips/'
label_dir = main_dir+'/filtered_label_chips/'

img_names = sorted(os.listdir(img_dir))
img_names.sort(key = lambda x: x.split('_',)[7])

label_names = sorted(os.listdir(label_dir))
label_names.sort(key = lambda x: x.split('_',)[7])

for ind, labl in enumerate(label_names):

    print("Working on label: ", str(labl))
    
    label = Image.open(label_dir+str(labl)).convert("L")

    
    label_array = np.asarray(label)
    label_array = label_array.flatten()
    orig_shape = label_array.shape
    
    new_array = np.zeros(len(label_array))
    
    for ind,val in enumerate(label_array):
            if val == 0:
                new_array[ind] = 1

            else:
                new_array[ind] = 0

    new_array = np.reshape(new_array, orig_shape)
