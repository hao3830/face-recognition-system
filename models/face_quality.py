import cv2
import onnxruntime as ort
import numpy as np

# TODO: Testing using google libary mediapipe facemesh
class FaceQuality:
    def __init__(self, backbone_path, quality_path, confident):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        cuda_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
            }
        # mem_limit = 1024 * 1024 * 1024
        # options.set_gpu_memory_limit(mem_limit)
        # ort.set_session_config(options, 'cuda', {"gpu_mem_limit": str(mem_limit)})
        self.backbone = ort.InferenceSession(backbone_path,options,providers=[ ("CUDAExecutionProvider", cuda_provider_options)])
        self.quality = ort.InferenceSession(quality_path, options, providers=[("CUDAExecutionProvider", cuda_provider_options)])
        self.confident = confident

        #Warm up
        print("Warm up model ...")
        self.warm_up()

    def predict(self, image):
        resized = cv2.resize(image, (112, 112))
        ccropped = resized[...,::-1] # BGR to RGB
        ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
        ccropped = np.reshape(ccropped, [1, 3, 112, 112])
        ccropped = np.array(ccropped, dtype = np.float16)
        ccropped = (ccropped - 127.5) / 128.0
        fc = self.backbone.run(None, {'input': ccropped})
        pred = self.quality.run(None, {'input': fc[0]})[0][0][0]

        if pred < self.confident:
            return "bad"
        
        return "good"

    def warm_up(self):
        rgb = np.random.randint(255, size=(112,112,3),dtype=np.uint8)
        ccropped = rgb[...,::-1] # BGR to RGB
        ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
        ccropped = np.reshape(ccropped, [1, 3, 112, 112])
        ccropped = np.array(ccropped, dtype = np.float16)
        ccropped = (ccropped - 127.5) / 128.0
        fc = self.backbone.run(None, {'input': ccropped})
        pred = self.quality.run(None, {'input': fc[0]})[0][0][0]