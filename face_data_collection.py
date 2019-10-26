# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data


import cv2
import numpy as np 

#Initialize Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = "./face_dataset/"

file_name = input("Enter the name of person : ")

while True:
	ret,frame = cap.read()

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
	if len(faces) == 0:
		continue

	k = 1

	faces = sorted(faces, key = lambda x : x[2]*x[3] , reverse = True)

	skip += 1

	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	for face in faces[:1]:
		x,y,w,h = face

		#Extract (Crop out the required face) : Region of Interest
		offset = 5
		face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_selection = cv2.resize(face_offset,(100,100))

		if skip % 10 == 0:
			face_data.append(face_selection)
			print (len(face_data))


		cv2.imshow(str(k), face_selection)
		k += 1
		
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

	cv2.imshow("faces",frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# Convert ur face list array into a numpy array
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print (face_data.shape)

# Save this data into file system
np.save(dataset_path + file_name, face_data)
print ("Dataset saved at : {}".format(dataset_path + file_name + '.npy'))

cap.release()
cv2.destroyAllWindows()