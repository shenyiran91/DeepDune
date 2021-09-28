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

#function to make a binary raster from an input image 
img_dir = '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/'


print('----Raster Data----')

shpfile = '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/labels/shapefiles/dune_fields_simp_cyl.shp'
image = img_dir+'themis_merged.tif'

img = rasterio.open(image)
shp = geopandas.read_file(shpfile)


print('bounds: ', img.bounds)
print('transform: ', img.transform)

CRS = shp.crs
transform = img.transform

binary_mask = img.read_masks(1)

print('original image crs: ', img.crs)

masked_dataset = rasterio.open(
    '/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/binary_themis_merge.tif',
    'w',
    driver='GTiff',
    height = binary_mask.shape[0],
    width = binary_mask.shape[1],
    count = 1,
    dtype = binary_mask.dtype,
    crs = CRS,
    transform = transform
)

print('raster crs: ', img.crs)
print('----End of Raster Data----')


masked_dataset.write(binary_mask,1)

print('masked image crs: ', masked_dataset.crs)


