import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')


p = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna','Mindy Kaling']

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

resized_images = []
for img in features:
    resized_img = cv.resize(img, (100, 100))  # Adjust the dimensions as needed
    resized_images.append(resized_img)
features = np.array(resized_images)

face_recognizer = cv.face.EigenFaceRecognizer_create()
face_recognizer.train(features, labels)

img = cv.imread(r'C:\Users\menia\Desktop\WinIT\opencv\Photo\val\ben_afflek\image.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('person', gray)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1 , 4)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h , x:x+h]
    resized_faces_roi = cv.resize(faces_roi, (100, 100))  # Resize input image to match training image size
    label, confidence = face_recognizer.predict(resized_faces_roi)
    print(f'label={p[label]} with a confidence of {confidence}')
    cv.putText(img,str(p[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('detected faces', img)

cv.waitKey(0)




