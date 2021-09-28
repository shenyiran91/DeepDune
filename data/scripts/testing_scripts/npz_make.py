import numpy as np
import os
from PIL import Image
from tempfile import TemporaryFile

def make_npz(data_folder, output_folder):

        img_dir = data_folder+'/filtered_image_tiles/'
        label_dir = data_folder+'/filtered_label_tiles/'
        
        image_list = []
        label_list = []

        #Finds and sort all files in datafolder subfolders (image_tiles and label_tiles)
        img_names = sorted(os.listdir(img_dir))
        img_names.sort(key = lambda x: x.split('_',)[2])
        print('img_names: ', img_names)

        
        label_names = sorted(os.listdir(label_dir))
        label_names.sort(key = lambda x: x.split('_',)[2])
        print('label_names: ', label_names)
        
        for ind,img in enumerate(img_names):

                #Opens images as Image objects.
                image = Image.open(img_dir+img).convert('L')
                label = Image.open(label_dir+label_names[ind])

                #Prints the size/shape of the image and label to ensure they are the same.
                print("-----------")
                print("Image size: ", image.size)
                print("Label size: ", label.size)

                if image.size == label.size:
                        print("Image and label sizes are consistent.")
                else:
                        print("Size inconsistency detected.")

                #Creates an array of pixel values from the image and label.
                img_array = np.asarray(image)
                label_array = np.asarray(label)
                
                #add each 2-D array to a list
                image_list.append(img_array)
                label_list.append(label_array)

        #convert lists to numpy arrays, yielding 2 arrays eaching containg 2-D arrays
        img_data = np.asarray(image_list)
        label_data = np.asarray(label_list)

                
        #img and label data are combined using the np.savez method
        print("img array shape: ", img_array.shape)
        print("label array shape: ", label_array.shape)
        
        #data = np.hstack((img_data,label_data))

        
        training_dataset = TemporaryFile()
        #np.savez(os.path(output_folder),data)
        np.savez(output_dir+'/training_data',img_data=img_data,label_data=label_data)
        
        return

data_dir = "/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles"
output_dir = "/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/training_data"
training_data = make_npz(data_dir, output_dir)

