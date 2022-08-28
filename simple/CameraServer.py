import ctypes
import io
import platform
import threading

import cv2
import flask
from flask import make_response
from flask import send_file

# 缓存
frameBuffer = None


# linux 释放内存
def malloc_trim():
    if platform.system().lower() == 'linux':
        ctypes.CDLL('libc.so.6').malloc_trim(0)


# 工作线程
# def worker():
#     print('start thread')
#     while (True):
#         if frameBuffer is not None:
#             cv2.waitKey(1)
#             cv2.imshow('frame', frameBuffer)
# threading.Thread(target=worker).start()


def capture_worker():
    print('capture_worker...')
    cap = cv2.VideoCapture(0)
    global frameBuffer
    while cap.isOpened():  # 当视频被打开时：
        ret, frame = cap.read()  # 读取视频，读取到的某一帧存储到frame，若是读取成功，ret为True，反之为False
        if ret:
            frameBuffer = frame


threading.Thread(target=capture_worker).start()

# 启动服务
server = flask.Flask(__name__)


@server.route('/capture', methods=['get', 'post'])
def capture():
    image = cv2.imencode('.jpg', frameBuffer)[1]
    response = make_response(send_file(
        io.BytesIO(image.tobytes()),
        mimetype='image/jpeg'
    ))
    # 释放内存
    malloc_trim()
    return response


server.run(debug=False, port=9090, host='0.0.0.0')  # 指定端口,host,0.0.0.0代表不管几个网卡，任何ip都可访问
