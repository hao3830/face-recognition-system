import cv2
import onnxruntime as ort
import numpy as np
class FaceQuality:
    def __init__(self, backbone_path, quality_path, confident):
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 3
        self.backbone = ort.InferenceSession(backbone_path)
        self.quality = ort.InferenceSession(quality_path)
        self.confident = confident

    def predict(self, image):
        resized = cv2.resize(image, (112, 112))
        ccropped = resized[...,::-1] # BGR to RGB
        ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
        ccropped = np.reshape(ccropped, [1, 3, 112, 112])
        ccropped = np.array(ccropped, dtype = np.float32)
        ccropped = (ccropped - 127.5) / 128.0
        fc = self.backbone.run(None, {'input': ccropped})
        pred = self.quality.run(None, {'input': fc[0]})[0][0][0]

        if pred < self.confident:
            return "bad"
        
        return "good"