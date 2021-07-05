import matplotlib.pylab as plt
import numpy as np
import cv2

img=cv2.imread("Road.jpg")
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

print(img.shape)
h=img.shape[0]
w=img.shape[1]


roi_v=[(0,h),(w/2,(h+100)/2),(w,h)]

def roi(img,v):
    mask=np.zeros_like(img)
    count=img.shape[2]
    mask_col=(255,)*count
    cv2.fillPoly(mask, v ,mask_col)
    m_img=cv2.bitwise_and(img, mask)
    return m_img


crop=roi(img,np.array([roi_v],np.int32),)

plt.imshow(crop)
plt.show() 