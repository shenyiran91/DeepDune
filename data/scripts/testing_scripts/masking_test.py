import numpy as np
import cv2
import shapely
from PIL import Image
import os
import geopandas
from pyspatialml import Raster
import copy 
import matplotlib.pyplot as plt
import rasterio

#function to make a copy of the original image, overlay the label polygons, and make a binary raster                                                                                                          
#def label_dune_fields(image,polygons):                                                                                                                                                                      
img_dir = '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/'
label_dir = '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/labels/shapefiles/'

#image processes
print('----Raster Data----')
image = img_dir+'binary_mask_test.tif'
#image = img_dir+'Mars_MO_THEMIS-IR-Day_mosaic_global_100m_v12_mars2000.tif'
#image = img_dir+'dayir_100mpix_full_mosaic.pgm'

img = rasterio.open(image)
print('raster crs: ', img.crs)
print('----End of Raster Data----')

img_data = Raster(img)

#polygons processes
print('----Polygon Data----')

shapefile_poly = label_dir+'dune_field_test_SimpleCylindrical.shp'   
dune_fields = geopandas.read_file(shapefile_poly)


if img.crs == dune_fields.crs:
    print("COORDINATE SYTEMS ARE CONSISTENT")

dune_mask = dune_fields.iloc[0:1,:]
print('polygon crs: ', dune_fields.crs)



print('----End of Polygon Data----')

#masking
masked_img = img_data.mask(dune_mask)


#fig, ax = plt.subplots(figsize=(9, 9))

#ax.set_aspect('equal')
#masked_img.plot()


plt.plot(masked_img)
#dune_fields.plot()

#dune_fields.plot()

#c = plt.imshow(binary_mask)

plt.show()

#extraction processes
#df_polygons = binary.extract_vector(dune_fields, return_array=True)

#testing
#print(img.shape)
#print(blank_image.shape)
#print(thresh.shape)
#cv2.imwrite('/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/binary.jpg',thresh)    
    
    
    
    
