import cv2
from rescale import rescaleframe,image_resize,scale_model
import numpy as np



def transform_video(vid,face_mesh,haar,human_face,breed_list,model,stframe,record,out,i):
    prevTime = 0

    while vid.isOpened():
        i +=1
        ret, frame = vid.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        face_count = 0
        # if results.multi_face_landmarks:
        #     for face_landmarks in results.multi_face_landmarks:
        #         face_count += 1
        #         mp_drawing.draw_landmarks(
        #         image = frame,
        #         landmark_list=face_landmarks,
        #         connections=mp_face_mesh.FACEMESH_CONTOURS,
        #         landmark_drawing_spec=drawing_spec,
        #         connection_drawing_spec=drawing_spec)
        # currTime = time.time()
        # fps = 1 / (currTime - prevTime)
        # prevTime = currTime

        fr = rescaleframe(frame)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        threshold,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
        face = haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
        h_face = human_face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
        for (x,y,w,h) in face:
            print(type(frame))
            cv2.rectangle(frame,(round(x/0.3),round(y/0.3)),
                        (round((x+w)/0.3),round((y+h)/0.3)),(0,255,0),thickness=2)
            breed = breed_list[np.argmax(model.predict(np.expand_dims(scale_model(frame),0)))] #predicting the breed on this frame
            cv2.putText(frame,breed,(round(x/0.3),round(y/0.3)),cv2.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2) #writing the breed name if cat is detected
            #cv.putText(frame,'Cat',(round(x/0.3),round(y/0.3)),cv.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2)
        for (x,y,w,h) in h_face:
            cv2.rectangle(frame,(round(x/0.3),round(y/0.3)),
                        (round((x+w)/0.3),round((y+h)/0.3)),(0,0,255),thickness=2)
            cv2.putText(frame,'Hooman',(round(x/0.3),round(y/0.3)),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2) # Creating a rectange with human mention if human is detected
        if record:
            #st.checkbox("Recording", value=True)
            out.write(frame)
        #Dashboard
        # kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
        # kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        # kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

        frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
        frame = image_resize(image = frame, width = 640)
        stframe.image(frame,channels = 'BGR',use_column_width=True)

def transform_image(face_mesh,image,haar,human_face,breed_list,model,st):

    results = face_mesh.process(image)
    out_image = image.copy()
    fr = rescaleframe(out_image)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    threshold,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    face = haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
    h_face = human_face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
    for (x,y,w,h) in face:
        print(type(out_image))
        cv2.rectangle(out_image,(round(x/0.3),round(y/0.3)),
        (round((x+w)/0.3),round((y+h)/0.3)),(0,255,0),thickness=2)
        breed = breed_list[np.argmax(model.predict(np.expand_dims(scale_model(out_image),0)))] # predictiong breed
        cv2.putText(out_image,breed,(round(x/0.3),round(y/0.3)),cv2.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2) # printing the predicted breed on the picture
        #cv.putText(out_image,'Cat',(round(x/0.3),round(y/0.3)),cv.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2)
    for (x,y,w,h) in h_face:
        cv2.rectangle(out_image,(round(x/0.3),round(y/0.3)),
        (round((x+w)/0.3),round((y+h)/0.3)),(0,0,255),thickness=2)
        cv2.putText(out_image,'Hooman',(round(x/0.3),round(y/0.3)),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)

    # for face_landmarks in results.multi_face_landmarks:
    #     face_count += 1

        #print('face_landmarks:', face_landmarks)

        # mp_drawing.draw_landmarks(
        # image=out_image,
        # landmark_list=face_landmarks,
        # connections=mp_face_mesh.FACEMESH_CONTOURS,
        # landmark_drawing_spec=drawing_spec,
        # connection_drawing_spec=drawing_spec)
        # kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
    st.subheader('Output Image')
    st.image(out_image,use_column_width= True)
