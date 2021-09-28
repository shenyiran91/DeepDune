from PIL import Image             # Python Imaging Library
import numpy as np                # Numerical Python 
import matplotlib.pyplot as plt   # Python plotting

image = "/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles/image_tiles/image_slice_29_12.png"
label = "/Users/cole/Dropbox/Courses/machine_learning/Machine-Learners/class_project/data/images/tiles/label_tiles/label_slice_29_12.png"


I = Image.open(image).convert('L')
L = Image.open(label)

print("I", I.size, I.mode, I.format)
print("L", L.size, L.mode, L.format)


print(L.histogram())
#I.show()
#L.show()

plt.imshow(np.array(I), cmap='gray')
plt.imshow(np.array(L), cmap='gray')

plt.show()
