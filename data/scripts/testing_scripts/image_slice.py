'''Script to slice large PNGs into smaller ones for analysis and place them into a subdirectory'''

import image_slicer
import os
from PIL import Image

def slice_images(dirname):

    img_dir = dirname+'/tiles/unclipped_images/'
    label_dir = dirname+'/tiles/unclipped_labels/'

    img_names = sorted(os.listdir(img_dir))
    img_names.sort(key = lambda x: x.split('_',)[7])

    print(img_names)
    label_names = sorted(os.listdir(label_dir))
    label_names.sort(key = lambda x: x.split('_',)[4])


    print('Slicing images and labels now...')
    
    for ind,img in enumerate(img_names):

        img_tiles = image_slicer.slice(img_dir+img, 1000, save=False)
        image_slicer.save_tiles(img_tiles, directory = dirname+'/tiles/image_tiles/', prefix='image_slice',format='png')

        label_tiles = image_slicer.slice(label_dir+label_names[ind], 1000, save=False)
        image_slicer.save_tiles(label_tiles, directory = dirname+'/tiles/label_tiles/', prefix='label_slice',format='png')

    print('Images and labels have been sliced successfully and stored.')
    
    return

directory = '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images'

slice_images(directory)

