#OpenCV Library
import cv2
#Glob module to find all the pathnames matching a specified pattern
import glob
import random
import math
# NumPy library for large, multi-dimensional arrays and matrices
import numpy as np
import dlib
import itertools
# scikit-learn library
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
import sys
#Detector for detecting human face from an image
detector = dlib.get_frontal_face_detector()
#Argument passing path of the model created during training
path = (sys.argv[1])
#Argument passing-
#If it is'F', it is real-time emotion analysis; otherwise, then static image checking #
image_flag = str(sys.argv[2])
#If it is static, argument passing the image path
img_path = sys.argv[3]
#Array of 8 emotions
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
#Using Adaptive Historam Equalization technique as a preprocessing measure to improve background contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#shape predictor for obtaining set of point locations of facial landmarks
landmarks_file='shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(landmarks_file)
def get_face_landmarks(image):
	detections = detector(image, 1)
    #Obtaining landmarks for an image
	for k,d in enumerate(detections):
        #Draw Facial Landmarks with the predictor class
		shape = predictor(image, d)
        # Create arrays for storing the X coordinate of landmark
		xlist = []
        # Create arrays for storing the Y coordinate of landmark
		ylist = []
        #For each 68 points from 'shape_predictor_68_face_landmarks.dat' file
		for i in range(1,68):
            #Store X and Y coordinates in two lists
			xlist.append(float(shape.part(i).x))
			ylist.append(float(shape.part(i).y))
        #Find the mean value of X coordinates
		xmean = np.mean(xlist)
        #Find the mean value of Y coordinates
		ymean = np.mean(ylist)
		x_centre = [(x-xmean) for x in xlist]
		y_centre = [(y-ymean) for y in ylist]
        #Create an array of landmarks
		landmarks = []
        #Iterate over x_centre , y_centre, xlist and ylist
		for x, y, w, z in zip(x_centre, y_centre, xlist, ylist):
			landmarks.append(w)
			landmarks.append(z)
            # Create np array of mean
			np_mean = np.asarray((ymean,xmean))
            # Create np array of coordinate points
			np_coordinates = np.asarray((z,w))
            #Compute the norm of the matrix
			dist = np.linalg.norm(np_coordinates-np_mean)
            # Append the norm to the landmarks matrix
			landmarks.append(dist)
			landmarks.append((math.atan2(y, x)*360)/(2*math.pi))
	return landmarks
#Function to predict the emotion from the trained model

def predict_result(img,model,image_flag='F'):
	# Try exception in case face captured or not within the window frame
	try :
		s=get_face_landmarks(img)
		res = model.predict_proba([s])
		# If the image flag is 'F',i.e., real time emotion detection
		if image_flag == 'F' :
			#Return the emotion with the maximum accuracy
			return (emotions[(res[0].argmax())])        # If the image flag is 'T', i.e., static image emotion recognition
		else:
		#Return the emotion with the maximum accuracy
			return (emotions[(res[0].argmax())])
	except :
		return [0]
#Main function

def main_try(image_flag='F'):
	# Load the model created during training
	model = joblib.load(path)
	#If image_flag is'F', execute the real-time emotion analysis
	if image_flag == 'F':
		#Open the default camera, i.e., webcam
		webcam = cv2.VideoCapture(0)
		# Capture the frame
		ret, frame = webcam.read()
		#Set the font type, positon of font to print, font color
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (30,300)
		fontScale              = 1
		fontColor              = (255,255,255)
		lineType               = 2
		bottomLeftCornerOfText1 = (50,300)
		color = [[255 ,0 ,0],[0,255,0]]
		# Loop until  a key 'q' is not pressed
		while ret:
			# If a key 'q' is pressed, exit the program
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			#Convert the frames into grayscale
			img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#Write the type of detected emotion in the frame
			cv2.putText(img_gray,'Emotion : {} '.format(predict_result(img_gray,model)),
					bottomLeftCornerOfText, 
					font, 
					fontScale,
					fontColor,
					lineType)
			#Display each frames
			cv2.imshow('frame',img_gray)
			# Capture frame-by-frame
			ret, frame = webcam.read()
		#Release the capture of webcam
		webcam.release()
		# Close all the windows and deallocate any associated memory usage
		cv2.destroyAllWindows()
 	# If image flag is 'T', predict the emotion of the image passed as command line argument
	else :
		#Read the image
		img_gray = cv2.imread(img_path, 0)
		#Predict the image from the trained model
		print(predict_result(img_gray,model,image_flag))

main_try(image_flag)

