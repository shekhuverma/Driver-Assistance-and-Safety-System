import cv2
import time,os,sys
import numpy as np
import concurrent.futures
from threading import Timer, Event


##cap = cv2.VideoCapture("I:\\Minor_Project\\Workflow\\demo\\video.mp4")

cap = cv2.VideoCapture("I://Minor_Project//Workflow//demo//video_1.mp4")


ret,frame=cap.read()
height,width = frame.shape[:2]

mask=np.full_like(frame,255)
mask[:340,:]=0
mask[height-60:height,:]=0
mask=mask[:,:,2]

half=width/2

#BL,BR,TL,TR
# coordinates={"BL":[0,height],"BR":[width,height],"TL":[0,340],"TR":[width,340]}

coordinates=np.array(([[[0,height-40],[width,height-40],[0,340],[width,340]]]),dtype="int32")

coordinates2=np.copy(coordinates)

i=1
#To save the images
def capture(frame_list):

	for l,frame in enumerate(frame_list, start=1):
		cv2.imwrite((str(i)+"_"+str(l)+str(".jpg")),frame)
	

def roi(image,coordinates):
	mask=np.zeros(image.shape,dtype='uint8')

	cv2.fillPoly(mask,coordinates,255)
	masked_image = cv2.bitwise_and(image,mask)
	return masked_image


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
					if coordinates[0][2][0]<=x:
						coordinates[0][2][0]=x

					if coordinates[0][0][0]<x1:
						coordinates[0][0][0]=x1

				if m>0: #right side line
					if coordinates[0][3][0]>=x:
						coordinates[0][3][0]=x

					if coordinates[0][1][0]>x1:
						coordinates[0][1][0]=x1

				# cv2.line(image,(x,320),(x1,height),[0,255,0],2)

				# image=cv2.circle(image,(x,320),50,(255,255,255),5)
				# image=cv2.circle(image,(x1,height),50,(255,255,255),5)
			else:
				continue
	except TypeError as  e:
		# print(e)
		pass


def reset():
	if not done.is_set():
		print("Clearing coordinates")
		coordinates=coordinates2
		Timer(10, reset).start()

done = Event()

Timer(10, reset).start()


if __name__=='__main__':
#Loop to acess the video feed
	while True:
		last_time=time.time()
		ret,frame=cap.read()
		frame_final_output=np.copy(frame)
		frame_cvt=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		frame_cvt2=np.copy(frame_cvt)
		frame_cvt3=cv2.bitwise_and(frame_cvt,frame_cvt,mask=mask)
		frame_cvt4=np.copy(frame_cvt3)
		frame_cvt=cv2.GaussianBlur(frame_cvt3,(3,3),0)
		edges = cv2.Canny(frame_cvt,86,270) #found out using hit and try using the sliders , need some fine tuning
		# edges = cv2.Canny(frame_cvt,90,180)
		#these parameters need tu be fine tuned
		lines = cv2.HoughLinesP(edges,10,np.pi/180,100,np.array([]),170,5)

		draw_lines(frame_cvt,lines,frame_cvt2)
		for a in coordinates[0]:
			image=cv2.circle(frame_cvt4,(a[0],a[1]),25,(255,255,255),5)


		temp=roi(edges,coordinates)
		
		contours,hierarchy = cv2.findContours(temp,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

		contour_list = []
		for contour in contours:
			approx = cv2.approxPolyDP(contour,0.02*cv2.arcLength(contour,True),True)
			area = cv2.contourArea(contour)
			if ((len(approx) >10 ) & (len(approx) < 30) & (area > 20) ):
				contour_list.append(contour)

		for a in contour_list:
			x,y,w,h = cv2.boundingRect(a)
			cv2.rectangle(frame_final_output,(x,y),(x+w,y+h),(255,255,255),2)

		cv2.drawContours(frame_cvt2, contour_list,  -1, (255,0,0), 2)

		cv2.imshow("frame_final_output",frame_final_output)
		fps=1.0/(time.time()-last_time)
		print("FPS =",fps)
		last_time=time.time()
		frame_list=[frame,frame_cvt3,edges,frame_cvt4,temp,frame_cvt2,frame_final_output]
		key=cv2.waitKey(25)
		if key == ord('q'):
			cv2.destroyAllWindows()
			break
		elif key==ord("j"):
			capture(frame_list)
			i+=1

	done.set()

