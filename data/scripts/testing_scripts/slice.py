from image_slicer import slice
from PIL import Image
import cv2
import numpy as np

img = Image.open('binary_themis_merge.tif')

img = img.convert('RGBA')
pixdata = img.load()

print(type(pixdata))
black_pixels = 0
non_black_pixels = 0

pixel_list = []

non_zero_pixels = []

for y in range(img.size[1]):

    i = 0
    
    for x in range(img.size[0]):
        
        if pixdata[x,y][0] == 0 and pixdata[x,y][1] == 0:
            
            black_pixels = black_pixels + 1

            #pixdata[x,y] = (255,255,255,0)
            
        else:
            pixel_list.append(pixdata[x,y])
            non_black_pixels = non_black_pixels + 1
            i = i + 1

    print(i)        
    non_zero_pixels.append(i)

non_zero_pixel_array = np.asarray(non_zero_pixels)

max_val = np.max(non_zero_pixel_array)

columns = max_val
row = len(non_zero_pixel_array)

print('number of black pixels: ', black_pixels)
print('number of non-black pixels: ', non_black_pixels)

pixel_array = np.asarray(pixel_list)
print(len(pixel_array))


#img = np.array(pixdata)

#image = Image.fromarray(img)

#print(type(image))
#img = cv2.cvtColor(pixdata, cv2.COLOR_BGR2RGB)

#cropped_img = np.array(pixdata)

#cv2.imwrite('/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/images/test_image.jpg',img)

#slice('I09599006RDR_b9.eq.jpg', 4)
