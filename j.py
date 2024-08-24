import cv2
import os
import numpy as np
from time import time
from plyer import notification
import pickle

# Directory where your images are stored
image_dir = 'E:/Opencv/data/photo'
database_dir = 'E:/Opencv/data/database_dir'

# Create directories if they don't exist
if not os.path.exists(database_dir):
    os.makedirs(database_dir)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create an LBPH face recognizer
recognizer = cv2.face_LBPHFaceRecognizer.create()

timings = 0

sos = 0

f = open("model.pkl","rb")
output = pickle.load(f)
f.close()


# Prepare training data from your photo collection
faces, labels = output

print(faces,labels)
recognizer.train(faces, np.array(labels))

print("Model training complete.")

# Initialize webcam capture
capture = cv2.VideoCapture(0)

face_detected_time = None
recognized_label = None
temp  = 0
notify = 0
counter  = 0
while True:
    ret, frame = capture.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rect = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        face = gray_frame[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)
        # print(f'label - {label} and confidebce {confidence}')
        # qprint(confidence)
        if confidence < 60:  # Adjust confidence threshold based on your needs
            
            if(notify == temp):
                counter += 1
            if(counter > 10):
                counter = 0
                notify = 0
                temp = 0
            print(f"{temp} - temp , {notify} - notify, {counter} - counter")
            print(label)
            if label == 1:  # Assuming label 0 is your face
                cv2.putText(frame, "Approved", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                recognized_label = "approved"
                temp = notify
            else:
                recognized_label = "intruder"
                cv2.putText(frame, "Red Intruder", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            recognized_label = "intruder"
            cv2.putText(frame, "Red Intruder", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if recognized_label == "intruder":
            if(sos == 2):
                print("SOSOSOKSJISXJSJ")
                cv2.imwrite("E:/Opencv/data/intrupic/1.png",frame)
                cv2.imshow("Bitch",frame)
                sos = 0
            if((notify > 55)):
                notify = 0
                if((time() - timings > 5)):
                    notification.notify(title="SOS",message="Some one is watching your screen")
                    timings  = time()
                    sos += 1

            if face_detected_time is None:
                face_detected_time = time()
            elif time() - face_detected_time > 5:
                cv2.putText(frame, "Red Intruder", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            notify += 1
        else:
            face_detected_time = None

        
        print(notify)

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
