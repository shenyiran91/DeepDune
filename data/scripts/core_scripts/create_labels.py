import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

''' Functions used for labeling, taken from https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/ '''

# create a mapping of tags to integers given the loaded mapping file
def create_tag_mapping(mapping_csv):
	# create a set of all known tags
	labels = set()
	for i in range(len(mapping_csv)):
		# convert spaced separated tags into an array of tags
		tags = mapping_csv['tags'][i].split(' ')
		# add tags to the set of known labels
		labels.update(tags)
	# convert set of labels to a list to list
	labels = list(labels)
	# order set alphabetically
	labels.sort()
	# dict that maps labels to integers, and the reverse
	labels_map = {labels[i]:i for i in range(len(labels))}
	inv_labels_map = {i:labels[i] for i in range(len(labels))}
	return labels_map, inv_labels_map

# create a mapping of filename to tags
def create_file_mapping(mapping_csv):
	file_mapping = dict()
	for i in range(len(mapping_csv)):
		name, tags = mapping_csv['image_name'][i], mapping_csv['tags'][i]
		file_mapping[name] = tags.split(' ')
	return file_mapping

    # create a one hot encoding for one list of tags
def one_hot_encode(tags, mapping):
	# create empty vector
	encoding = zeros(len(mapping), dtype='uint8')
	# mark 1 for each tag in the vector
	for tag in tags:
		encoding[mapping[tag]] = 1
	return encoding
 
# load all images into memory
def load_dataset(path, file_mapping, tag_mapping):
	photos, targets = list(), list()

        # enumerate files in the directory
        img_names = sorted(os.listdir(img_dir))
        img_names.sort(key = lambda x: x.split('_',)[7])

        label_names = sorted(os.listdir(label_dir))
        label_names.sort(key = lambda x: x.split('_',)[4])

        for filename in os.listdir(img_names):
		# load image
		photo = load_img(path + filename, target_size=(128,128))
		# convert to numpy array
		photo = img_to_array(photo, dtype='uint8')
		# get tags
		tags = file_mapping[filename[:-4]]
		# one hot encode tags
		target = one_hot_encode(tags, tag_mapping)
		# store
		photos.append(photo)
		targets.append(target)
	X = asarray(photos, dtype='uint8')
	y = asarray(targets, dtype='uint8')
	return X, y

#perform tagging below using the above functions
main_dir = '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles'
img_dir = main_dir+'/filtered_image_tiles_test_gs/'
label_dir = main_dir+'/filtered_label_tiles_test_gs/'

#Finds and sort all files in datafolder subfolders (image_tiles and label_tiles)                                                                                                                                                            
img_names = sorted(os.listdir(img_dir))
img_names.sort(key = lambda x: x.split('_',)[7])

label_names = sorted(os.listdir(label_dir))
label_names.sort(key = lambda x: x.split('_',)[4])

#construct the csv used for labeling
label_img_list = []
labels = []
for index,image in enumerate(label_names):

    label_img_list.append(str(image))
    labels.append('dune not_dune')

label_img_array = np.asarray(label_img_list)
labels_array = np.asarray(labels)

df = pd.DataFrame()

df['image_name'] = pd.Series(label_img_array)
df['tags'] = pd.Series(labels_array)

df.to_csv('/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/training_data/train.csv', header=True)
mapping_csv = pd.read_csv('/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/training_data/train.csv')

# load the mapping file
filename = '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/training_data/train.csv'
mapping_csv = pd.read_csv(filename)
# create a mapping of tags to integers
tag_mapping, _ = create_tag_mapping(mapping_csv)
# create a mapping of filenames to tag lists
file_mapping = create_file_mapping(mapping_csv)
print(file_mapping)
# load the jpeg images
folder = img_dir
X, y = load_dataset(folder, file_mapping, tag_mapping)
print(X.shape, y.shape)
# save both arrays to one file in compressed format
savez_compressed('/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/training_data/training_data_test.npz', data=X, labels=y)
