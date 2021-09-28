import numpy as np
import os
from PIL import Image
from tempfile import TemporaryFile
from tqdm import tqdm

def make_npz(data_folder, output_folder):

        img_dir = data_folder+'/image_chips/'
        label_dir = data_folder+'/label_chips/'
        
        image_list = []
        label_list = []

        #Finds and sort all files in datafolder subfolders (image_tiles and label_tiles)
        img_names = sorted(os.listdir(img_dir))
        img_names.sort(key = lambda x: x.split('_',)[7])
        
        label_names = sorted(os.listdir(label_dir))
        label_names.sort(key = lambda x: x.split('_',)[7])
        
        for ind,img in enumerate(tqdm(img_names)):
                
                #Opens images as Image objects.
                image = Image.open(img_dir+img)
                label = Image.open(label_dir+label_names[ind])

                #Prints the size/shape of the image and label to ensure they are the same.
                #print("-----------")
                #print("Image mode: ", image.mode)
                #print("Label mode: ", label.mode)

                
                #if image.size == label.size:
                        #print("Image and label sizes are consistent.")
               #else:
                        #print("Size inconsistency detected.")
        
                #Creates an array of pixel values from the image and label.
                img_array = np.asarray(image)
                label_array = np.asarray(label)

                #make label image just 0s where theres no label and 1s where there is. There's gotta be a better way to do this
                #orig_shape = label_array.shape
                #label_array = label_array.flatten()
                #new_array = np.zeros(len(label_array))

                #for ind,val in enumerate(label_array):
                #        if val == 0:
                #                new_array[ind] = 1

                #        else:
                #                new_array[ind] = 0

                #new_array = np.reshape(new_array, orig_shape)

                #add each 2-D array to a list
                #if new_array.shape == (256,256):
                if label_array.shape == (256,256):       
                        image_list.append(img_array)
                        label_list.append(label_array)
                        #print("Final label image shape: ", new_array.shape)
                        #print("Final data image shape: ", img_array.shape)

                #else:
                        #print("This image has different dimensions: ", new_array.shape," and has been excluded.")
                              
        #convert lists to numpy arrays, yielding 2 arrays eaching containg 2-D arrays
        img_data = np.asarray(image_list, dtype=np.float64)
        label_data = np.asarray(label_list,dtype=np.float64)

        print(img_data.shape)
        print(label_data.shape)
        
        training_dataset = TemporaryFile()
        np.savez(output_dir+'/mars_dunes_full_dataset',data=img_data,labels=label_data)
        
        return

data_dir = "/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles"
output_dir = "/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/training_data"
training_data = make_npz(data_dir, output_dir)

