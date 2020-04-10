import cv2
import time,os
import numpy as np
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

frame=cv2.imread("demo.jpg")
frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

h,s,v=cv2.split(frame_hsv)

height,width = frame.shape[:2]
mask=np.full_like(frame,255)
mask[:520,:]=0
mask=mask[:,:,2]

frame_hsv = cv2.bitwise_and(frame_hsv,frame_hsv,mask=mask)

print("The mean of hue = ",np.mean(h))
print("The SD of hue = ",np.std(h))

hist_h=cv2.calcHist([frame_hsv],[0],None,[179],[3,179])
hist_s=cv2.calcHist([frame_hsv],[1],None,[255],[3,255])
hist_v=cv2.calcHist([frame_hsv],[2],None,[255],[3,255])

print("Frame shape of the entire image =" ,frame.shape)

print("Frame shape of red channel =" ,frame[:,:,2].shape)


#HSV histogram

f1=plt.figure(1)
plt.plot(hist_h)
plt.plot(np.mean(frame_hsv[0]),"g",label="Mean")
plt.plot(np.std(frame_hsv[0]),"r",label="SD")
plt.grid(True)
plt.legend(title="Legend")
plt.title("HSV colour space HUE channel")
# plt.show()

f2=plt.figure(2)
plt.plot(hist_s)
plt.grid(True)
plt.title("HSV colour space SAT channel")
# plt.show()

f3=plt.figure(3)
plt.plot(hist_v)
plt.grid(True)
plt.title("HSV colour space VAL channel")

f4=plt.figure(4)
plt.imshow(frame_hsv)


plt.show()


#histogram range(203-205) 207936

# frame_cvt=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
# # frame_cvt = cv2.blur(frame_cvt,(7,7))

# #HSV histogram equalazation

# h,s,v=cv2.split(frame_cvt)
# h1 = cv2.blur(h,(5,5))
# # h=cv2.equalizeHist(h)
# # s=cv2.equalizeHist(s)
# # v=cv2.equalizeHist(v)
# frame_cvt=cv2.merge((h,s,v))
# frame_cvt = cv2.bitwise_and(frame_cvt,frame_cvt,mask=mask) #To mask the sky
# cv2.imshow("frame_cvt",frame_cvt)
 

cv2.imshow("frame_hsv",frame_hsv)
cv2.imshow("frame",frame)
cv2.waitKey()


