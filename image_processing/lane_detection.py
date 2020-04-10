import cv2
import time,os,winsound
import numpy as np


import threading

def coordinates_reset():
  threading.Timer(5.0, printit).start()
  print "Hello, World!"

printit()


kernel = (1/159)*(np.array([(2,4,5,4,2),(4,9,12,9,4),(5,12,15,12,5),(4,9,12,9,4),(2,4,5,4,2)],dtype="float"))

kernel1 = np.ones((3,3),np.uint8)
##cap = cv2.VideoCapture("I:\\Minor_Project\\Workflow\\demo\\video.mp4")

cap = cv2.VideoCapture("I://Minor_Project//Workflow//demo//video_1.mp4")


ret,frame=cap.read()
height,width = frame.shape[:2]
mask=np.full_like(frame,255)
mask[:340,:]=0
mask=mask[:,:,2]

half=width/2

#BL,BR,TL,TR
# coordinates={"BL":[0,height],"BR":[width,height],"TL":[0,340],"TR":[width,340]}

coordinates=np.array(([0,height],[width,height],[0,340],[width,340]),dtype="int")
def slope(x1,y1,x2,y2): 
	return ((y2-y1)/(x2-x1))

def draw_lines(image,lines,image2):
	try:
		for line in lines:
			coords=line[0]
			m=slope(coords[0],coords[1],coords[2],coords[3])
			if m==0:
				return
			if (coords[1] and coords[3])>345:
				try:
					x=int(coords[0]+((340-coords[1])/m)) #y=0
					x1=int(coords[0]+((height-coords[1])/m)) #y=height
					if x<0 or x1<0:
						return
					if x>width or x1>width:
						return	
				except OverflowError as e:
					return
				if m<0: #Left side line
					if coordinates[2][0]<=x:
						coordinates[2][0]=x

					if coordinates[0][0]<x1:
						coordinates[0][0]=x1

				if m>0: #right side line
					if coordinates[3][0]>=x:
						coordinates[3][0]=x

					if coordinates[1][0]>x1:
						coordinates[1][0]=x1

				cv2.line(image,(x,320),(x1,height),[0,255,0],2)

				# image=cv2.circle(image,(x,320),50,(255,255,255),5)
				# image=cv2.circle(image,(x1,height),50,(255,255,255),5)
			else:
				continue
	except TypeError as  e:
		# print(e)
		pass

#Loop to acess the video feed
while True:
	last_time=time.time()
	ret,frame=cap.read()
	frame_cvt=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	frame_cvt2=np.copy(frame_cvt)
	frame_cvt=cv2.bitwise_and(frame_cvt,frame_cvt,mask=mask)
	# blur=cv2.GaussianBlur(frame_cvt,(7,7),0)
	edges = cv2.Canny(frame_cvt,86,270) #found out using hit and try using the sliders , need some fine tuning
	#these parameters need tu be fine tuned
	# edges=cv2.GaussianBlur(edges,(7,7),0)
	# edges = cv2.dilate(edges,kernel1,iterations = 1)
	lines = cv2.HoughLinesP(edges,10,np.pi/180,100,np.array([]),170,5)
	
	draw_lines(frame_cvt,lines,frame_cvt2)
	# print(coordinates)
	for a in coordinates:
		image=cv2.circle(frame_cvt,(a[0],a[1]),25,(255,255,255),5)
		
	cv2.imshow("frame_cvt",frame_cvt)
	# cv2.imshow("frame_cvt2",frame_cvt2)
	# cv2.imshow("Edges",edges)
	fps=1.0/(time.time()-last_time)
	print("FPS =",fps)
	last_time=time.time()
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break

