import numpy as np
import os
from PIL import Image
from tempfile import TemporaryFile
from tqdm import tqdm

''' This is a condensed version of npz_make.py which only exports the full image tile/chip dataset to .npz - Label files are not included here. '''

def make_npz(data_folder, output_folder):

        img_dir = data_folder+'/unclipped_images/'
        
        image_list = []

        #Finds and sort all files in datafolder subfolders (image_tiles)
        img_names = sorted(os.listdir(img_dir))
        img_names.sort(key = lambda x: x.split('_',)[7])


        img2 = '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles/unclipped_images/Mars_MO_THEMIS-IR-Day_mosaic_global_100m_v12_clip-01-02.png'

        image = Image.open(img).convert("L")

        img_array = np.asarray(image)

        np.savez(output_dir+'/mars_dunes_01-02',data=img_array)


        
        
        
        i = 0
        
#        for ind,img in enumerate(img_names):
#                #Opens images as Image objects.
#                image = Image.open(img_dir+img)

                #Creates an array of pixel values from the image and label.
#                img_array = np.asarray(image)
#                             
#                else:
                        #convert lists to numpy arrays, yielding 2 arrays eaching containg 2-D arrays
#                        img_data = np.asarray(image_list, dtype=np.float64)
#                        training_dataset = TemporaryFile()
                        #np.savez(output_dir+'/mars_dunes_full_ds_'+str(i),data=img_data)

#                        image_list = []

#                        i = i + 1
                        
        return

data_dir = "/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles"
output_dir = "/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/training_data"
training_data = make_npz(data_dir, output_dir)

