'''Script to slice large PNGs into smaller ones for analysis and place them into a subdirectory'''

import image_slicer
import os
from PIL import Image
import numpy as np

def slice_images(dirname):

    img_dir = dirname+'/tiles/unclipped_images/'
    label_dir = dirname+'/tiles/unclipped_labels_test/'

    img_names = sorted(os.listdir(img_dir))
    img_names.sort(key = lambda x: x.split('_',)[7])

    label_names = sorted(os.listdir(label_dir))
    label_names.sort(key = lambda x: x.split('_',)[4])


    print('Slicing images and labels now...')
    
    for ind,img in enumerate(img_names):

        im = Image.open(img_dir+str(img)).convert("L")
        lab = Image.open(label_dir+str(label_names[ind])).convert("L")

        im_array = np.asarray(im)
        lab_array = np.asarray(lab)

        M = 256
        N = 256
                         
        img_tiles = [im_array[x:x+M, y:y+N] for x in range(0, im_array.shape[0],M) for y in range(0,im_array.shape[1],N)]
        label_tiles = [lab_array[x:x+M, y:y+N] for x in range(0, lab_array.shape[0],M) for y in range(0,lab_array.shape[1],N)]
        image_array = np.asarray(img_tiles)
        label_array = np.asarray(label_tiles)

        for index,tile in enumerate(image_array):
            pil_image = Image.fromarray(tile)
            pil_image.save('/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles/image_chips/'+str(img)[:-4]+"-"+str(index)+".png", format="png")

        for index,label in enumerate(label_array):
            pil_label = Image.fromarray(label)
            pil_label.save('/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles/label_chips/'+str(img)[:-4]+"-"+str(index)+"_label.png", format="png")

        print("Done slicing image: ", str(img), 'and label: ', str(label_names[ind]))
    
    print('Images and labels have been sliced successfully and stored.')
    
    return

directory = '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images'

slice_images(directory)

