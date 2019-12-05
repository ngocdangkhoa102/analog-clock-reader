import cv2 as cv
import numpy as np 

def nothing():
	pass

src = cv.imread("images/anh4.png")
# src = cv.imread("clock-rgb.png")
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

minVal = 0;
maxVal = 255;

# cv.namedWindow('Threshold Value')
cv.namedWindow('Panel')
cv.createTrackbar('lo_val','Panel',minVal,maxVal,nothing)
cv.createTrackbar('hi_val','Panel',minVal,maxVal,nothing)
cv.createTrackbar('type','Panel',0,3,nothing)

# im = src[:,0:src.shape[1] - val]
while True:
	lo_val = cv.getTrackbarPos('lo_val','Panel')
	hi_val = cv.getTrackbarPos('hi_val','Panel')
	c = cv.getTrackbarPos('type','Panel')
	if c == 0:
		imbin = gray[:]
	elif c == 1:
		ret,imbin = cv.threshold(gray,lo_val,hi_val,cv.THRESH_BINARY)	
	elif c == 2:
		imbin = cv.adaptiveThreshold(gray, hi_val, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 61, 12)
	else:
		imbni = gray[:]
	# cv.imshow('orginal',src)	
	cv.imshow('thresholded',imbin)
	k = cv.waitKey(1) & 0xFF
	if k == 27:
		cv.imwrite('imbin.png',imbin)
		break
cv.destroyAllWindows()
