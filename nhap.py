import cv2 as cv
import numpy as np 
import math
import time
import urllib.request

def norm2(x0,y0,x1,y1):
	return math.sqrt((x0-x1)**2+(y0-y1)**2)

def change_axis(x0,y0,x,y):
	X = x - x0
	Y = y - y0
	return X,Y

def equation_solver(x0,y0,x1,y1):
	a = (y0 - y1)/(x0 - x1)
	b = y0 - a*x0
	return a,b

def get_alpha(a):
	if a < 0:
		alpha = np.pi - np.arctan(-a)
	else:
		alpha = np.arctan(a) 
	return alpha

def getxy(x0,y0,x1,y1,ox,oy):
	if norm2(ox,oy,x0,y0) < norm2(ox,oy,x1,y1):
		return x1,y1
	else:
		return x0,y0

def get_theta(a,x,y):
	tmp = get_alpha(a)*180/np.pi	
	if x< 0 and y < 0:
		theta_m = 270 + tmp
	elif x < 0:
		theta_m = 90 + tmp
	elif y < 0:
		theta_m = tmp - 90
	else:
		theta_m = tmp + 90
	return theta_m

def get_value(hour_hand,min_hand):
	xm,ym = min_hand
	xh,yh = hour_hand
	am,bm = equation_solver(0,0,xm,ym)
	ah,bh = equation_solver(0,0,xh,yh)
	valOfmin = np.round(get_theta(am,xm,ym)/6).astype("int")
	valOfhour = np.round(get_theta(ah,xh,yh)/30).astype("int")
	if valOfmin >= 25:
		valOfhour = valOfhour - 1
	return(valOfhour,valOfmin)

def getab(lines):
	output = np.zeros((lines.shape[0],3),dtype = np.float32)
	index = 0
	for line in lines:
		x0,y0,x1,y1 = line[0]
		output[index,0],output[index,1] = equation_solver(x0,y0,x1,y1)
		output[index,2] = index
		index = index + 1
	return output

def check_line(lines,ox,oy):
	index = 0
	output = lines[:]
	end_time = time.time() + 3
	while time.time() < end_time:
		if index >= output.shape[0]:
			break
		x0,y0,x1,y1 = output[index][0]
		if norm2(x0,y0,ox,oy) > 25 and norm2(x1,y1,ox,oy) > 25:
			output = np.delete(output,[index],0)
			index = index - 1
		index = index + 1
	return output

def filter(mtx):
	output = mtx[:]
	index = 0
	end_time = time.time() + 3
	while time.time() < end_time:
		if output.shape[0] == 2:
	 		break
		c = 1
		for line in output[index+1:]:
			if abs(get_alpha(line[0]) - get_alpha(output[index,0])) < 12*np.pi/180:
				c = 0
				break
		if c == 1:	
			index = index + 1
		else:
			output = np.delete(output,[index],0)
	return output

src = cv.imread("images/anh (15).png")
# src = cv.imread("clock/clock (4).png")
# src = cv.imread("test-fail/fail (11).png")
cv.imshow("Src",src)
cv.waitKey(0)

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)

lo_val = 61
hi_val = 255
imbin = cv.adaptiveThreshold(gray, hi_val, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 61, 12)
cv.imshow("Threshold",imbin)
cv.waitKey(0)
edges = cv.Canny(imbin,220,250,apertureSize = 3)
cv.imshow("Edge",edges)
cv.waitKey(0)

circles = cv.HoughCircles(edges,cv.HOUGH_GRADIENT,1,20,param1=50,param2=40,minRadius=5,maxRadius=100)
circles = np.round(circles[0,:]).astype("int")

for circle in circles:
	(x,y,r) = circle
	if norm2(x,y,src.shape[1]/2,src.shape[0]/2) < 20:
		break

cv.circle(src,(x,y),r,(0,255,0),4)
cv.imshow('Result',src)
cv.waitKey(0)

clockim = imbin[y-r:y+r,x-r:x+r]
clockim = 255 - clockim

kernel1 = np.ones((1,1),dtype = np.uint8)
kernel2 = np.ones((2,2),dtype = np.uint8)
kernel3 = np.ones((3,3),dtype = np.uint8)
kernel4 = np.ones((4,4),dtype = np.uint8)
for index in range(0,2):
	# clockim = cv.erode(clockim,kernel1)
	clockim = cv.erode(clockim,kernel2)
	# clockim = cv.erode(clockim,kernel3)
for index in range(0,1):
	clockim = cv.dilate(clockim,kernel2)
cv.imshow("clock-only",clockim)
cv.waitKey(0)

offset = 55
clock_pos = (int(clockim.shape[0]/2) - offset,int(clockim.shape[0]/2) + offset, int(clockim.shape[1]/2) - offset,int(clockim.shape[1]/2) + offset)
(y0,y1,x0,x1) = clock_pos
tmp = clockim[y0:y1,x0:x1]
cv.imshow("Tmp",tmp)
edges = cv.Canny(tmp,40,200,apertureSize = 3)
cv.imshow("edges",edges)
lines = cv.HoughLinesP(edges,1,np.pi/180,22,minLineLength = 25,maxLineGap = 7)
# lines = cv.HoughLinesP(edges,1,np.pi/180,23,minLineLength = 25,maxLineGap = 7) #for anh 14
# print(lines.shape[0])
# print('stage 1')

# tst = np.zeros(edges.shape,dtype = np.uint8)
# print(tst.shape)
# ox,oy = (int(tst.shape[1]/2),int(tst.shape[0]/2))
# print(ox,oy)
# for line in lines:
# 	x0,y0,x1,y1 = line[0]
# 	# print(x0,y0,x1,y1)
# 	cv.line(tst,(x0,y0),(x1,y1),255,1)
# 	cv.imshow("clock with line",tst)
# 	cv.waitKey(0)

tst = np.zeros(edges.shape,dtype = np.uint8)
ox,oy = (int(tst.shape[1]/2),int(tst.shape[0]/2))

if lines is not None:
	lines = check_line(lines,ox,oy)
	min_hand = (0,0)
	ab_mtx = getab(lines)
	ab_mtx = filter(ab_mtx)

	if ab_mtx.shape[0] == 2:
		for index in range(0,2):
			x0,y0,x1,y1 = lines[int(ab_mtx[index,2])][0]
			x1,y1 = getxy(x0,y0,x1,y1,ox,oy)
			X1,Y1 = change_axis(ox,oy,x1,y1)
			if norm2(0,0,X1,Y1) > norm2(0,0,min_hand[0],min_hand[1]):
				hour_hand = min_hand
				min_hand = X1,Y1
			else:
				hour_hand = X1,Y1
			cv.line(tst,(ox,oy),(x1,y1),255,1)
			# cv.rectangle(tst,(x1,y1),(x1+5,y1+5),255,1)
		cv.imshow("clock with line",tst)
		# cv.waitKey(0)

	# print(min_hand)
	# print(hour_hand)

		hours, minutes = get_value(hour_hand,min_hand)
		# cv.waitKey(0)

		from google_speech import Speech 
		text = "It's " + str(hours) + ":" + str(minutes) + "..."
		lang = "en"
	else:
		from google_speech import Speech 
		text = "Can't see any clock"
		lang = "en"
else:
	from google_speech import Speech 
	text = "Can't see any clock"
	lang = "en"
print(text)
speech = Speech(text,lang)
speech.play()

cv.waitKey(0)
cv.destroyAllWindows()
