
import cv2
import os
import numpy as np
from time import time
from plyer import notification
import pickle

image_dir = 'E:/Opencv/data/photo'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []

    for dir_name in dirs:
        label = int(dir_name)
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_rect = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces_rect:
                faces.append(gray_image[y:y+w, x:x+h])
                labels.append(label)
    
    return faces, labels

# Prepare training data from your photo collection
faces, labels = prepare_training_data(image_dir)

ouput = [faces,labels]

f = open('model.pkl',"wb")

pickle.dump(ouput,f)

f.close()

print("Model Stored succ")