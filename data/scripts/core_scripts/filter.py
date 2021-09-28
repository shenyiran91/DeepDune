import numpy as np
import os
from PIL import Image
from shutil import copyfile

main_dir = '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles'
img_dir = main_dir+'/image_chips'
label_dir = main_dir+'/label_chips'

#Finds and sort all files in datafolder subfolders (image_tiles and label_tiles)                                                              
img_names = sorted(os.listdir(img_dir))

img_names.sort(key = lambda x: x.split('_',)[7])

label_names = sorted(os.listdir(label_dir))
label_names.sort(key = lambda x: x.split('_',)[7])

def get_color(file):
    img = Image.open(file).convert('L')
    colors = Image.Image.getcolors(img)
    return colors

no_label = []
is_label = []

colors_list = []

for ind,img in enumerate(label_names):
 
    print("=========================")
    print("Now filtering image ", str(img_names[ind]), 'to find tiles with lables corresponding to ', str(label_names[ind]))
    colors = get_color(label_dir+'/'+str(img))

    if len(colors) > 1 and colors[0][0] > (256**2)*0.05:
        print(colors[0][0])
        colors_list.append(colors)
        #this indicates that there are pixels corresponding to labels, and that labeled pixels account for more than 10% of the image. We copy both the image and the labels into a filtered directory
        copyfile(label_dir+'/'+img, main_dir+'/filtered_labeled_chips_5percent/'+img)
        copyfile(img_dir+'/'+img_names[ind], main_dir+'/filtered_image_chips_5percent/'+img_names[ind])

        is_label.append(img)
    
    else:
        no_label.append(img)       

color_array = np.asarray(colors_list)

np.sort(color_array, axis=0)


for i,val in enumerate(color_array):
    print(val)
        
print("Total labeled tiles: ", len(is_label))
print("Total tiles: ", len(img_names))
          
