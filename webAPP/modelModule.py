import tensorflow as tf
from PIL import Image
import base64
import io
import numpy as np
from cv2 import *
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
class cnnTF:
    def __init__(self):
        self.dims = (256, 256, 3)
        self.model = tf.keras.models.load_model('maskModel2.h5')
        self.font = FONT_HERSHEY_SIMPLEX

    def b64Pred(self, b64String):
        print("starting")
        base64_decoded = base64.b64decode(b64String.encode('utf-8'))
        image_rgba = Image.open(io.BytesIO(base64_decoded))
        image_rgb = image_rgba.convert('RGB')
        image_np = np.array(image_rgb)
        image_np = np.expand_dims(image_np, axis=0)
        print("finished processing")
        preds = self.model.predict(image_np)
        # 0: With mask ; 1: without mask
        return preds[0][0]

    def launchFramePreds(self):
        cap = VideoCapture(0)
        cap.set(CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(CAP_PROP_FRAME_HEIGHT, 1080)
        for x in range(200):
            print("Iter is ", x)
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            resized = resize(frame, (256, 256), interpolation = INTER_AREA)
            RGB = cvtColor(resized, COLOR_BGR2RGB)

            np_resizedRGB = np.array(RGB)
            np_resizedRGB = np.expand_dims(np_resizedRGB, axis=0)

            preds = self.model.predict(np_resizedRGB)
            print(preds[0][0])
            message = ""
            color = (0,0,0)
            if round(preds[0][0]) == 0:
                message = "Wearing Mask"
                color = (0,255,0)
            else:
                message = "Not Wearing Mask"
                color = (0,0,255)

            # Display the resulting frame
            putText(resized,
                        message,
                        (0,25),
                        self.font, 1,
                        color,
                        2,
                        cv2.LINE_4)
            imshow('frame', resized)
            # time.sleep(0.1)
            if waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        destroyAllWindows()


#testObj = cnnTF()
#testObj.launchFramePreds()
