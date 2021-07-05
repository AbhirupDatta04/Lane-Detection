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
    #count=img.shape[2]
    mask_col=(255)
    cv2.fillPoly(mask, v ,mask_col)
    m_img=cv2.bitwise_and(img, mask)
    return m_img

def linedraw(img,lines):
    img=np.copy(img)
    lineimg=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    for l in lines:
        for x1,y1,x2,y2 in l:
            cv2.line(lineimg,(x1,y1),(x2,y2),(100,255,0),thickness=5)
            
    img=cv2.addWeighted(img,0.8,lineimg,1,0.0)
    return img

grayimg=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
canimg=cv2.Canny(grayimg,100,200)
crop=roi(canimg,np.array([roi_v],np.int32),)
lines=cv2.HoughLinesP(crop, rho=6, theta=np.pi/60, threshold=160 ,lines=np.array([]),minLineLength=40,maxLineGap=25)



imgwithlines=linedraw(img, lines)
plt.imshow(imgwithlines)
plt.show() 