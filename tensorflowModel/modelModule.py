import tensorflow as tf
from PIL import Image
import base64
import io
import numpy as np
import cv2




class cnnTF:
    def __init__(self):
        self.dims = (256, 256, 3)
        self.model = tf.keras.models.load_model('maskModel.h5')
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def b64Pred(self, b64String):
        base64_decoded = base64.b64decode(b64String)
        image_rgba = Image.open(io.BytesIO(base64_decoded))
        image_rgb = image_rgba.convert('RGB')
        image_np = np.array(image_rgb)
        image_np = np.expand_dims(image_np, axis=0)

        preds = self.model.predict(image_np)
        # 0: With mask ; 1: without mask
        return preds[0][0]

    def launchFramePreds(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            resized = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_AREA)
            RGB = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            np_resizedRGB = np.array(RGB)
            np_resizedRGB = np.expand_dims(np_resizedRGB, axis=0)

            preds = self.model.predict(np_resizedRGB)
            print(preds[0][0])
            message = ""
            if round(preds[0][0]) == 0:
                message = "Wearing Mask"
            else:
                message = "Not Wearing Mask"

            # Display the resulting frame
            cv2.putText(resized,
                        message,
                        (50, 50),
                        self.font, 1,
                        (0, 255, 255),
                        2,
                        cv2.LINE_4)
            cv2.imshow('frame', resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


testObj = cnnTF()
testObj.launchFramePreds()
