import cv2 as cv
from rescale import rescaleframe,scale_model
from model_creation import model_20, open_image
import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# img = cv.imread('catim.jpg')

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
def transform_video(video):
    haar = cv.CascadeClassifier('haar_cat.xml')
    human_face =cv.CascadeClassifier('haar_human.xml')


    model = model_20()
    model.load_weights('model/')
    breed_list = list(pd.read_csv('breed.csv')['0'])
    # for (x,y,w,h) in face:
    #     cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

    # print(face)
    # print(y)
    # cv.imshow('cat',img)
    # cv.waitKey(0)

    capture = cv.VideoCapture(video)

    while True :
        isTrue,frame = capture.read()
        fr = rescaleframe(frame)
        gray = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
        threshold,thresh = cv.threshold(gray,150,255,cv.THRESH_BINARY)
        face = haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
        h_face = human_face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
        for (x,y,w,h) in face:
            print(type(frame))
            cv.rectangle(frame,(round(x/0.3),round(y/0.3)),
                        (round((x+w)/0.3),round((y+h)/0.3)),(0,255,0),thickness=2)
            breed = breed_list[np.argmax(model.predict(np.expand_dims(scale_model(frame),0)))]
            cv.putText(frame,breed,(round(x/0.3),round(y/0.3)),cv.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2)
            #cv.putText(frame,'Cat',(round(x/0.3),round(y/0.3)),cv.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2)
        for (x,y,w,h) in h_face:
            cv.rectangle(frame,(round(x/0.3),round(y/0.3)),
                        (round((x+w)/0.3),round((y+h)/0.3)),(0,0,255),thickness=2)
            cv.putText(frame,'Hooman',(round(x/0.3),round(y/0.3)),cv.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)

        # for (x,y,w,h), (a,b,c,d) in zip(face, h_face):
        #     cv.rectangle(frame,(round(x/0.3),round(y/0.3)),
        #                  (round((x+w)/0.3),round((y+h)/0.3)),(0,255,0),thickness=2)
        #     cv.rectangle(frame,(round(a/0.3),round(b/0.3)),
        #                  (round((a+c)/0.3),round((b+d)/0.3)),(0,0,255),thickness=2)


        cv.imshow('Video',frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()
