import argparse
from flask import Flask
import multiprocessing as mp
from app import AppService
from infer import InferService

mp.set_start_method("fork")

# Get configs
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="if GPU is present")
parser.add_argument("--checkpt", default="./result", help="path to folder containing the model")
ARGS = parser.parse_args()
gpu_flag = ARGS.gpu
ckpt_path = ARGS.checkpt

# data channels for IPC between services
infer_q = mp.Queue()
res_q = mp.Queue()

# Run services
app = Flask("AppService")
app_service = AppService(app, infer_q, res_q)
infer_service = InferService(ckpt_path, gpu_flag, infer_q, res_q)
p_app = mp.Process(target=app_service.run)
p_infer = mp.Process(target=infer_service.infer)
p_app.start()
p_infer.start()
p_app.join()
p_infer.join()
