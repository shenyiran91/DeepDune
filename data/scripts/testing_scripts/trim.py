import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('I09599006RDR_b9.eq.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[-1]

cnt_list = []

for index,contour in enumerate(contours):
    if len(contour) > 100 and len(contour) < 50000:
        print(contour)
        print("length of contour ",len(contour))
        print("------------------")
        cnt_list.append(contour)
        print("index is ", index, "out of ", len(contours))
        
img = cv2.drawContours(img,contours,1183,(0,255,0),3)

cv2.imshow("contours",img)
cv2.waitKey()

crop = img[box]


#plt.imshow(im,'gray')

#plt.show()
cv2.imwrite('test.jpg',img)
