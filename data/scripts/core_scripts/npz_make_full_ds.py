import numpy as np
import os
from PIL import Image
from tempfile import TemporaryFile
from tqdm import tqdm

''' This is a condensed version of npz_make.py which only exports the full image tile/chip dataset to .npz - Label files are not included here. '''

def make_npz(data_folder, output_folder):

        img_dir = data_folder+'/image_chips/'
        
        image_list = []

        #Finds and sort all files in datafolder subfolders (image_tiles)
        img_names = sorted(os.listdir(img_dir))
        img_names.sort(key = lambda x: x.split('_',)[7])
        print(len(img_names))
        
        i = 0
        
        for ind,img in enumerate(img_names):
                
                if (len(image_list) < 2000):
                        #Opens images as Image objects.
                        image = Image.open(img_dir+img)

                        #Creates an array of pixel values from the image and label.
                        img_array = np.asarray(image)
                
                        #add each 2-D array to a list
                        if img_array.shape == (256,256):
                                image_list.append(img_array)
                                print(ind)
                        else:
                                print("the end of the row is after: ", ind, " images, when we encounter an image that is ", img_array.shape)
                                                
                else:
                        #convert lists to numpy arrays, yielding 2 arrays eaching containg 2-D arrays
                        img_data = np.asarray(image_list, dtype=np.float64)
                        training_dataset = TemporaryFile()
                        #np.savez(output_dir+'/mars_dunes_full_ds_'+str(i),data=img_data)

                        image_list = []

                        i = i + 1
                        
        return

data_dir = "/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles"
output_dir = "/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/training_data"
training_data = make_npz(data_dir, output_dir)

