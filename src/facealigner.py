# import the necessary packages
from helpers import FACIAL_LANDMARKS_IDXS
from helpers import model_points
from helpers import shape_to_np
from helpers import eye_aspect_ratio
import numpy as np
import cv2
import math

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
#EYE_AR_THRESH = 0.3
#EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
#COUNTER = 0
#TOTAL = 0

class FaceAligner:
	def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
		desiredFaceWidth=256, desiredFaceHeight=None, EYE_AR_THRESH = 0.2,
		EYE_AR_CONSEC_FRAMES = 3, COUNTER = 0, TOTAL = 0):
		# store the facial landmark predictor, desired output left
		# eye position, and desired output face width + height
		self.predictor = predictor
		self.desiredLeftEye = desiredLeftEye
		self.desiredFaceWidth = desiredFaceWidth
		self.desiredFaceHeight = desiredFaceHeight
		self.EYE_AR_THRESH = EYE_AR_THRESH
		self.EYE_AR_CONSEC_FRAMES = EYE_AR_CONSEC_FRAMES
		self.COUNTER = COUNTER
		self.TOTAL = TOTAL

		# if the desired face height is None, set it to be the
		# desired face width (normal behavior)
		if self.desiredFaceHeight is None:
			self.desiredFaceHeight = self.desiredFaceWidth

	def align(self, image, gray, rect):
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = self.predictor(gray, rect)
		shape = shape_to_np(shape)
		image_points = np.array([
		                            shape[30],     # Nose tip
		                            shape[8],      # Chin
		                            shape[36],     # Left eye left corner
		                            shape[45],     # Right eye right corne
		                            shape[48],     # Left Mouth corner
		                            shape[54]      # Right mouth corner
                                ], dtype="double")


		# extract the left and right eye (x, y)-coordinates
		(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
		leftEyePts = shape[lStart:lEnd]
		rightEyePts = shape[rStart:rEnd]
		
		leftEAR = eye_aspect_ratio(leftEyePts)
		rightEAR = eye_aspect_ratio(rightEyePts)
		
		ear = (leftEAR + rightEAR) / 2.0
		if ear < self.EYE_AR_THRESH:
		    self.COUNTER += 1
		    
		else:
		    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
		        self.TOTAL += 1
		    self.COUNTER = 0
		# compute the center of mass for each eye
		leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

		# compute the angle between the eye centroids
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180

		# compute the desired right eye x-coordinate based on the
		# desired x-coordinate of the left eye
		desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

		# determine the scale of the new resulting image by taking
		# the ratio of the distance between eyes in the *current*
		# image to the ratio of distance between eyes in the
		# *desired* image
		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist / dist

		# compute center (x, y)-coordinates (i.e., the median point)
		# between the two eyes in the input image
		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
			(leftEyeCenter[1] + rightEyeCenter[1]) // 2)

		# grab the rotation matrix for rotating and scaling the face
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

		# update the translation component of the matrix
		tX = self.desiredFaceWidth * 0.5
		tY = self.desiredFaceHeight * self.desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		# apply the affine transformation
		(w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
		#output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
		output = cv2.warpAffine(image, M, (w, h))
		# Camera internals
		size = image.shape
		focal_length = size[1]
		center = (size[1]/2, size[0]/2)
		camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )
        
        #print "Camera Matrix :\n {0}".format(camera_matrix)
		# return the aligned face
		dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
		(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
		
		#print ("Rotation Vector:\n {0}".format(rotation_vector))
		#print ("Translation Vector:\n {0}".format(translation_vector))
		R, _ = cv2.Rodrigues(rotation_vector)
		sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
		singular = sy < 1e-6   
		if not singular:
		    x = math.atan2(R[2,1] , R[2,2])
		    y = math.atan2(-R[2,0], sy)
		    z = math.atan2(R[1,0], R[0,0])
		else:
		    x = math.atan2(-R[1,2], R[1,1])
		    y = math.atan2(-R[2,0], sy)
		    z = 0		            
		euler = np.array([x, y, z])
		#pitch, yaw, roll
		     
		# Project a 3D point (0, 0, 1000.0) onto the image plane.
		# We use this to draw a line sticking out of the nose
		(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		p1 = ( int(image_points[0][0]), int(image_points[0][1]))
		p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
		return output, shape, p1, p2, euler, self.TOTAL, ear, leftEyePts, rightEyePts
	
	
