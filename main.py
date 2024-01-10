import streamlit as st
import cv2
from ultralytics import YOLO
import time
from pickle import load
from pathlib import Path
from pickle import dump

# Loading a model
# model = YOLO('yolov8n.pt')
# dump(model, open('yolo.pkl', 'wb'))

model_ = Path(__file__).parents[0] / 'yolo.pkl'

model = load(open(model_, 'rb'))

video = cv2.VideoCapture('http://195.196.36.242/mjpg/video.mjpg')
video2 = cv2.VideoCapture('http://77.222.181.11:8080/mjpg/video.mjpg')

threshold = 0.5


st.title("Real time object detection with YOLO V8 on multiple public RTSP")

tab1, tab2 = st.tabs(["Soltorget Pajala, Sweden", "Kaiskuru Skistadion, Norway"])

def cctv(video):
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

    FRAME_WINDOW = st.image([])

    while True:
        _,frame = video.read()
        ###########################
        ## resizing the window ##
        # width = 1080
        # height = 720
        # dim = (width, height)
        # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        ############################
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1,y1, x2, y2, score, class_id = result

        
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        

        FRAME_WINDOW.image(frame)

        ### To run it in the local window
        # cv2.imshow('RTSP', frame)
        # k = cv2.waitKey(1)
        
        # if btn_click == True:
        #     break

with tab1:
    btn_click = st.button("CCTV 1")
    if btn_click == True:
        cctv(video)


with tab2:
    btn_click = st.button("CCTV 2")
    if btn_click == True:
        cctv(video2)

# video.release()
# cv2.destroyAllWindows()