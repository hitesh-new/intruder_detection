import cv2
import os,sys


# Create the directory for storing the database if it doesn't exist
database_dir = 'E:/Opencv/data/photo/1'
if not os.path.exists(database_dir):
    os.makedirs(database_dir)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture object to use the webcam
capture = cv2.VideoCapture(0)

# Counter for the number of saved images
image_count = 2000

print("Press 'q' to quit and 's' to save the detected face.")

while True:
    # Capture frame-by-frame
    ret, frame = capture.read()
    if not ret:
        break

    # Convert the frame to grayscale (Haar cascade works with grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Wait for a key press and handle the key press
    key = cv2.waitKey(1) & 0xFF

    # if key == ord('s'):
        # Save the detected face(s)
    for (x, y, w, h) in faces:
        if(True):            # Crop and save each detected face   
            if(image_count == 2100):
                print(image_count)
                sys.exit()
            face = frame[y:y+h, x:x+w]
            face_filename = os.path.join(database_dir, f'{image_count}.png')
            if cv2.imwrite(face_filename, face) :
                print(f"saved{face_filename}")
        image_count += 1
    
    if key == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
