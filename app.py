from flask import Flask, json, request
import cv2
import numpy as np
from multiprocessing import Queue
import time
import threading

class AppService:
    def __init__(self, app: Flask, infer_q: Queue, res_q: Queue):
        self.app = app
        self.infer_q = infer_q
        self.res_q = res_q
        self.mutex = threading.Lock()
        self.next_id = 0

        # Register endpoints
        self.app.add_url_rule('/ping', view_func=self.ping)
        self.app.add_url_rule('/infer', view_func=self.infer, provide_automatic_options=True, methods=['POST'])

    def get_next_image_id(self):
        self.mutex.acquire()
        try:
            next_id = self.next_id
            self.next_id += 1
        finally:
            self.mutex.release()
        return next_id

    def run(self):
        self.app.run(port=4000, host='0.0.0.0')

    def ping(self):
        """
        To poll if the service is alive The response should be in JSON serialised and running
        """
        data = {'message': 'pong'}
        response = self.app.response_class(
            response=json.dumps(data),
            status=200,
        )
        return response

    def infer(self):
        msg = request.files['image']
        img_data = np.asarray(bytearray(msg.read()), dtype="uint8")
        image = cv2.imdecode(img_data, -1)

        id = self.get_next_image_id()
        print("processing id: ", id)
        self.infer_q.put((id, image))

        # Listen to res_q
        data = {}
        while True:
            print("getting item....")
            item = self.res_q.get()
            print("got item: ", item[0])
            if id != item[0]:
                print("putting back item in queue....")
                self.res_q.put(item)
                time.sleep(1)
            else:
                print("item MATCHHH")
                data['image'] = item[1]
                break
        response = self.app.response_class(
            response=json.dumps(data),
            status=200,
        )
        return response
