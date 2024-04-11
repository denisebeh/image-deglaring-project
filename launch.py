import argparse
from flask import Flask
import multiprocessing as mp
from app import AppService
from infer import InferService
import cv2
from utils import preprocess_data, utils

mp.set_start_method("fork")

# Get configs
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="GPU is present")
parser.add_argument("--debug", action="store_true", help="debug logging")
parser.add_argument("--checkpt", default="./result", help="path to folder containing the model")
ARGS = parser.parse_args()
gpu_flag = ARGS.gpu
debug_flag = ARGS.debug
ckpt_path = ARGS.checkpt

# data channels for IPC between services
infer_q = mp.Queue()
res_q = mp.Queue()

# Run services
app = Flask("AppService")
app_service = AppService(app, infer_q, res_q)
print("AppService initialized")
infer_service = InferService(ckpt_path, gpu_flag, debug_flag)
print("InferService initialized")
p_app = mp.Process(target=app_service.run)
p_app.start()

# Blocking get from queue until an item is available
while True:
    (id, image) = infer_q.get()
    print("Processing image id: ", id)
    image = preprocess_data.grayscale_image(image)
    tmp_all = utils.prepare_single_image(image)
    img = infer_service.predict(tmp_all)
    data = cv2.imencode('.png', img)
    res_q.put((id, data))
    print("Predicted image id: ", id)
