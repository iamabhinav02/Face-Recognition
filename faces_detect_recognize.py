import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('Cascade/data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle","rb") as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)


while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
	
	for x,y,w,h in faces:
		#print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]

		id_, conf = recognizer.predict(roi_gray)
		if conf>=25 and conf<=100:
			#print(id_)
			#print(labels[id_])
			name = labels[id_]
			font = cv2.FONT_HERSHEY_SIMPLEX
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		#cv2.imwrite("my-image.png",roi_color)

		color = (0,255,0) #BGR 0-255
		stroke = 2
		height = y + h
		width = x + w
		cv2.rectangle(frame, (x,y), (width,height), color, stroke)

		#eyes = eye_cascade.detectMultiScale(roi_gray)
		#for ex,ey,ew,eh in eyes:
			#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0.255,2),2)

	cv2.imshow('Live Stream..',frame)
	key = cv2.waitKey(20)
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()