import cv2
import time,os,winsound
import numpy as np

def nothing(x):
    pass

def beep():
    winsound.Beep(2500,50)

cv2.namedWindow('result')
kernel = np.ones((1,1), np.uint8) 


##cap = cv2.VideoCapture("I:\\Minor_Project\\Workflow\\demo\\video.mp4")
cap = cv2.VideoCapture("I://Minor_Project//Workflow//demo//video_1.mp4")



ret,frame=cap.read()
height,width = frame.shape[:2]
mask=np.full_like(frame,255)
mask[:320,:]=0
mask=mask[:,:,2]

cv2.createTrackbar('lower', 'result',0,800,nothing)
cv2.createTrackbar('upper', 'result',0,800,nothing)

#Loop to acess the video feed
while True:
	last_time=time.time()
	ret,frame=cap.read()
	frame_cvt=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# frame_cvt=cv2.equalizeHist(frame_cvt)
	bilateral=cv2.bilateralFilter(frame_cvt,9,75,75)
	# blur = cv2.blur(frame,(3,3))
	l = cv2.getTrackbarPos('lower','result')
	u = cv2.getTrackbarPos('upper','result')
	edges = cv2.Canny(bilateral,l,u) #found out using hit and try using the sliders , need some fine tuning 
	temp=cv2.bitwise_and(edges,edges,mask=mask)
	dilated=cv2.dilate(temp,kernel, iterations=1) 
	contours,hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	#Arguments-source image,contour retrieval mode,contour approximation method
	#Returns - 
	 
	# areas = [cv2.contourArea(c) for c in contours]
	# try:
	# 	max_index = np.argmax(areas)
	# except ValueError:
	# 	continue

	# cnt=contours[max_index] #Biggest contour
	# x,y,w,h = cv2.boundingRect(cnt)
	# cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

	#Printing all the contours to see all the pothole
	for a in contours:
		x,y,w,h = cv2.boundingRect(a)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)

	#cv2.drawContours(frame, contours, -1, (255,255,255), 3)
	# cv2.imshow("dilated",dilated)
	# cv2.imshow("output",frame)
	cv2.imshow("dilated",dilated)
	# cv2.imshow("bilateral",bilateral)
	cv2.imshow("frame_cvt",frame)
	# cv2.imshow("temp",temp)
	#cv2.imshow("edges",edges)
	print("FPS==",1.0/(time.time()-last_time))
	last_time=time.time()
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break

