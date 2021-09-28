import image_slicer
import os

dirname = '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images'

#images = os.listdir(dirname+'/images')


for root, dirs, files in os.walk(dirname, topdown = False):
    for img in files:
        print(img)
        tiles = image_slicer.slice(dirname+'/'+str(img), 500, save=False)
        print("Slicing tile ", img)
        image_slicer.save_tiles(tiles, directory = dirname+'/tiles/',prefix=str(img)[0:15]+'_slice',format='jpeg')
    
