import cv2
import time,os
import numpy as np
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

# def nothing(x):
# 	pass


# cv2.namedWindow('result',cv2.WINDOW_AUTOSIZE) #To create a blank window that resizes to the shape of the sliders


kernel = np.ones((7,7),np.uint8)


##cap = cv2.VideoCapture("I:\\Minor_Project\\Workflow\\demo\\video.mp4")
cap = cv2.VideoCapture("I://Minor_Project//Workflow//demo//video_1.mp4")


ret,frame=cap.read()
height,width = frame.shape[:2]
mask=np.full_like(frame,255)
mask[:320,:]=0
mask=mask[:,:,2]

# cv2.createTrackbar('saturation', 'result',0,255,nothing)
# cv2.createTrackbar('value', 'result',0,255,nothing)
# cv2.createTrackbar('hue', 'result',0,179,nothing)
# cv2.createTrackbar('saturation1', 'result',0,255,nothing)
# cv2.createTrackbar('value1', 'result',0,255,nothing)
# cv2.createTrackbar('hue1', 'result',0,179,nothing)



#Loop to acess the video feed
while True:
	ret,frame=cap.read()
	frame_cvt=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	# frame_cvt = cv2.blur(frame_cvt,(7,7))

	#HSV histogram equalazation
	h,s,v=cv2.split(frame_cvt)
	h = cv2.blur(h,(5,5))
	# h=cv2.equalizeHist(h)
	# s=cv2.equalizeHist(s)
	# v=cv2.equalizeHist(v)
	frame_cvt=cv2.merge((h,s,v))
	frame_cvt = cv2.bitwise_and(frame_cvt,frame_cvt,mask=mask) #To mask the sky


	# for a,b in enumerate(["h","s","v"]):
	# 	hist=cv2.calcHist([masked],[a],None,[256],[0,256])
	# 	plt.plot(hist)
	# 	plt.title(b)
	# 	plt.show()
	# 	plt.close()
	# frame = cv2.blur(frame,(3,3))
	# hist_h=cv2.calcHist(masked,[0],None,[179],[0,179])
	# hist_s=cv2.calcHist(masked,[1],None,[256],[0,255])
	# hist_v=cv2.calcHist(masked,[2],None,[256],[0,255])

	# s = cv2.getTrackbarPos('saturation','result')
	# v = cv2.getTrackbarPos('value','result')
	# h = cv2.getTrackbarPos('hue','result')
	# s1 =255-cv2.getTrackbarPos('saturation1','result')
	# v1 =255-cv2.getTrackbarPos('value1','result')
	# h1 =179-cv2.getTrackbarPos('hue1','result')

	# lower=np.array([h,s,v],dtype="uint8")
	# upper=np.array([h1,s1,v1],dtype="uint8")

	# print("lower =",lower,"upper =",upper)

	lower=np.array([130,50,50])
	upper=np.array([150,200,200)

	mask_final = cv2.inRange(frame_cvt,lower,upper)

	opening = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
	opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

	cv2.imshow("mask",mask_final)
	cv2.imshow("opening",opening)

	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(frame,frame, mask= mask_final)

	cv2.imshow("res",res)
	# cv2.imshow("frame",frame)
	# cv2.imshow("result")
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break

