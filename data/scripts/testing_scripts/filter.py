import numpy as np
import os
from PIL import Image
from shutil import copyfile

main_dir = '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles'
img_dir = main_dir+'/image_tiles'
label_dir = main_dir+'/label_tiles'

#Finds and sort all files in datafolder subfolders (image_tiles and label_tiles)                                                              
img_names = sorted(os.listdir(img_dir))
img_names.sort(key = lambda x: x.split('_',)[2])

label_names = sorted(os.listdir(label_dir))
label_names.sort(key = lambda x: x.split('_',)[2])

def get_color(file):
    img = Image.open(file)
    colors = Image.Image.getcolors(img)
    return colors

blank_list = []

#for root, dirs, files in os.walk(label_names, topdown = False):
for ind,img in enumerate(label_names):

        colors = get_color(label_dir+'/'+str(img))

        if len(colors) > 1:
            #this indicates there are pixels corresponding to labels. We copy both the image and the labels into a filtered directory
            copyfile(label_dir+'/'+img, main_dir+'/filtered_label_tiles/'+img)
            copyfile(img_dir+'/'+img_names[ind], main_dir+'/filtered_image_tiles/'+img_names[ind])

            blank_list.append(img)
            
        else:
            continue
        
print('total tiles that do not have labels: ', len(blank_list))
print('total tiles that do have labels: ', len(img_names)-len(blank_list))
