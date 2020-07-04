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

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


testObj = cnnTF()
testObj.launchFramePreds()
