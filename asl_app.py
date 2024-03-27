from src.backbone import TFLiteModel, get_model
from src.landmarks_extraction import mediapipe_detection, draw, extract_coordinates, load_json_file 
from src.config import SEQ_LEN, THRESH_HOLD
import numpy as np
import cv2
import time
import mediapipe as mp
import time
import cv2
import streamlit as st

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

s2p_map = {k.lower():v for k,v in load_json_file("src/sign_to_prediction_index_map.json").items()}
p2s_map = {v:k for k,v in load_json_file("src/sign_to_prediction_index_map.json").items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

models_path = [
    './models/islr-fp16-192-8-seed_all42-foldall-last.h5',
]
models = [get_model() for _ in models_path]

# Load weights from the weights file.
for model, path in zip(models, models_path):
  model.load_weights(path)


def real_time_asl():
      
    st.set_page_config(page_title="ASL Recognition")
    st.title("Real Time ASL Recognition")
    st.caption("Powered by EquiSpeak")  
    """
    Perform real-time ASL recognition and display a continuous video stream using OpenCV
    and Streamlit.
    """
    
    start_button_pressed = st.button("Start Recognition")

    res = []
    tflite_keras_model = TFLiteModel(islr_models=models)
    sequence_data = []
    
    if start_button_pressed:

        cap = cv2.VideoCapture(0)
        
        frame_placeholder = st.empty()
        stop_button_pressed = st.button("Stop")

        start = time.time()

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened() and not stop_button_pressed:

                ret, frame = cap.read()

                image, results = mediapipe_detection(frame, holistic)
                draw(image, results)

                try:
                    landmarks = extract_coordinates(results)
                except:
                    landmarks = np.zeros((468 + 21 + 33 + 21, 3))
                sequence_data.append(landmarks)

                sign = ""

                if len(sequence_data) % SEQ_LEN == 0:
                    prediction = tflite_keras_model(np.array(sequence_data, dtype=np.float32))["outputs"]
                    if np.max(prediction.numpy(), axis=-1) > THRESH_HOLD:
                        sign = np.argmax(prediction.numpy(), axis=-1)
                    sequence_data = []

                image = cv2.flip(image, 1)
                cv2.putText(image, f"{len(sequence_data)}", (3, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                image = cv2.flip(image, 1)

                if sign != "" and decoder(sign) not in res:
                    res.insert(0, decoder(sign))

                height, width = image.shape[0], image.shape[1]
                white_column = np.ones((height // 8, width, 3), dtype='uint8') * 255
                image = cv2.flip(image, 1)
                image = np.concatenate((white_column, image), axis=0)

                # Convert the image to RGB format for Streamlit (optional but recommended)
                frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                cv2.putText(frame, f"{', '.join(str(x) for x in res)}", (3, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)

                frame_placeholder.image(frame,channels="RGB")

                # Optional cleanup (consider if memory usage is a concern)
                # del frame  # Remove the frame after displaying it

                # Wait for a key to be pressed.
                if cv2.waitKey(10) & 0xFF == ord("q") or stop_button_pressed:
                    break

        cap.release()
        cv2.destroyAllWindows()


real_time_asl()
