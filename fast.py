#Modified by Augmented Startups 2021
#Face Landmark User Interface with StreamLit
#Watch Computer Vision Tutorials at www.augmentedstartups.info/YouTube
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
from catface.utils.rescale import rescaleframe,scale_model
from catface.ml.model_creation import model_20, open_image
import pandas as pd
from catface.utils.rescale import image_resize
from catface.utils.media_transformation import transform_video,transform_image

haar = cv2.CascadeClassifier('catface/face-detection/haar_cat.xml')
human_face =cv2.CascadeClassifier('catface/face-detection/haar_human.xml')

model = model_20()

model.load_weights('model/')
breed_list = list(pd.read_csv('data/breed.csv')['0'])

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_VIDEO = 'data/videos/cat_video.mp4'
DEMO_IMAGE = 'data/images/cat.jpg'

st.title('Face Mesh Application using MediaPipe')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Face Mesh Application using MediaPipe')
st.sidebar.subheader('Parameters')




app_mode = st.sidebar.selectbox('Choose the App mode',
['Run on Video','Run on Image','About App',]
)


if  app_mode =='Run on Video':

    #Running our detection model on a default video first and then the user has the choice to upload a video of its own or using a its webcam

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    #st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )
    # max faces
    # max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    # st.sidebar.markdown('---')
    # detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    # tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)


    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

    kpi1, kpi2, kpi3 = st.columns(3)

    # with kpi1:
    #     st.markdown("**FrameRate**")
    #     kpi1_text = st.markdown("0")

    # with kpi2:
    #     st.markdown("**Detected Faces**")
    #     kpi2_text = st.markdown("0")

    # with kpi3:
    #     st.markdown("**Image Width**")
    #     kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    with mp_face_mesh.FaceMesh(
    # min_detection_confidence=detection_confidence,
    # min_tracking_confidence=tracking_confidence ,
    # max_num_faces = max_faces
    ) as face_mesh:
        fr = transform_video(vid,face_mesh,haar,human_face,breed_list,model,stframe,record,out,i)

    st.text(fr)
    st.text('Video Processed')

    output_video = open('output1.mp4','rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out. release()


elif app_mode =='Run on Image':

    #Running our detection model on a default video first and then the user has the choice to upload a video of its own or using a its webcam


    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    st.markdown("**Detected Faces**")
    kpi1_text = st.markdown("0")
    st.markdown('---')

    # max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    # st.sidebar.markdown('---')
    # detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    # st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))


    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    face_count = 0
    # Dashboard
    with mp_face_mesh.FaceMesh(
    static_image_mode=True) as face_mesh:

        response =transform_image(face_mesh,image,haar,human_face,breed_list,model,st)
    st.sidebar.text(response)

elif  app_mode =='About App':
    st.markdown('In this application we are using **MediaPipe** for creating a Face Mesh. **StreamLit** is to create the Web Graphical User Interface (GUI) ')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.video('https://youtu.be/4JTb6HF9cUA?t=673')

    st.markdown('''
          # About Me \n
            Hey this is Abdellatif Hani from LeWagon. \n



            Also check us out on Social Media
            - [LinkedIn](https://www.linkedin.com/in/abdellatifhani/)
            - [Github](https://github.com/Abdl242)


            ''')



# Watch Tutorial at www.augmentedstartups.info/YouTube
