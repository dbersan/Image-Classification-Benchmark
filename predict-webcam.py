GPU = False
LIST_WEBCAMS=False
CAM_INDEX=0

import os
if not GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import tensorflow as tf
import numpy as np
import time

# Solution to weird bug on tensorflow
# Source: https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-629862126
if GPU:
    physical_devices = tf.config.experimental.list_physical_devices('GPU') 
    for physical_device in physical_devices: 
        tf.config.experimental.set_memory_growth(physical_device, True)

PRED_THRESH=0.4
MODEL_FOLDER='model'
IMG_SIZE = 150
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,450)
fontScale              = 1
fontColor              = (20,200,10)    
lineType               = 2
labels_str = ['bird', 'noclass', 'bottle', 'lion']

def returnCameraIndexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

# Get cameras available
if LIST_WEBCAMS:
    indices = []
    indices = returnCameraIndexes()
    print('WEBCAMS:', indices)
    exit(0)


def getPredict(model_folder):
    # Load model
    loaded_model = tf.keras.models.load_model('model') 

    # Predict function 
    predict = loaded_model.signatures["serving_default"]

    return predict 

# Converts image to correct size, range, color depth, etc...
def preProcessImage(img, img_size, invert_channels=True, normalize=True):

    # Convert to correct size
    height, width, c = img.shape
    start_col = int(max(width/2-height/2, 0))
    end_col = int(min(start_col+height, width))
    img = img[:, start_col:end_col, ... ]
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC )

    # Invert color channels
    if invert_channels:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize values
    if normalize:
        img = img/255.0

    # Adds 4th axis
    img = img[np.newaxis, ...]

    return img


predict = getPredict(MODEL_FOLDER)
last_time = time.time()

try:
    cv2.namedWindow("test")
    
    cam = cv2.VideoCapture(CAM_INDEX)

    while True:
        ret, img_bgr = cam.read()
        if not ret:
            print("failed to grab img_bgr")
            break

        # Calculate fps
        now = time.time()
        dt = now - last_time
        fps = 1.0/dt
        last_time = now

        # Pre process image
        img = preProcessImage(img_bgr, IMG_SIZE)

        prediced_label = ''

        
        # Predict
        prediction = predict(tf.constant(img, dtype=tf.float32))
        probabilities = prediction['output_layer'][0].numpy()
        predicted_index = np.argmax(probabilities)
        prediced_label = labels_str[predicted_index]
        predicted_prob = round(probabilities[predicted_index], 3)
        if predicted_prob < PRED_THRESH:
            prediced_label='noclass'

        # Write on image
        text = str(round(fps,1)) + ' '+prediced_label
        cv2.putText(img_bgr, text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        

        # Show image
        cv2.imshow("test", img_bgr)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    cam.release()
    cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("Keyboard interrupt, terminating")
    cam.release()
    cv2.destroyAllWindows()

except Exception as e:
    print("EXCEPTION OCCURRED")
    print(e)
    cam.release()
    cv2.destroyAllWindows()

