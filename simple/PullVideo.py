import os
import threading

import cv2
import paddlex as pdx
from paddlex.cv.models.utils.visualize import draw_bbox_mask
import paddle

# 工作空间
_workspace = os.path.join(os.path.dirname(__file__), '..')

# model = pdx.load_model(os.path.join(_workspace, 'best_model'))
useGpu = int(paddle.device.cuda.device_count()) > 0
print('Enable GPU: ' + str(useGpu))
predictor = pdx.deploy.Predictor(model_dir=os.path.join(_workspace, 'inference_model'), use_gpu=useGpu)

frameBuffer = None

import paddle

paddle.utils.run_check()


def worker():
    print('start thread')
    while (True):
        if frameBuffer is not None:
            cv2.waitKey(1)
            result = predictor.predict(frameBuffer)
            result = draw_bbox_mask(frameBuffer, result, threshold=0.5, color_map=None)
            cv2.imshow('frame', result)


cap = cv2.VideoCapture(
    'https://cmgw-vpc.lechange.com:8890/LCO/7K0058EPAZ449C2/0/1/20220610T023510/a969630330a1d150e3b775b6a36013c1.m3u8?proto=https')
threading.Thread(target=worker).start()
while cap.isOpened():  # 当视频被打开时：
    ret, frame = cap.read()  # 读取视频，读取到的某一帧存储到frame，若是读取成功，ret为True，反之为False
    frameBuffer = frame
cv2.destroyAllWindows()  # 关闭所有窗口

# cv2.imshow('frame', frame)
# key = cv2.waitKey(1000)  # 等待一段时间，并且检测键盘输入
# print(key)
# if ret:  # 若是读取成功
#     # cv2.imshow('frame', frame)  # 显示读取到的这一帧画面
#     # buffer = cv2.imdecode(np.array(frame, dtype=np.uint8), -1)
#     result = predictor.predict(frame)
#     # draw_bbox_mask
#     result = draw_bbox_mask(frame, result, threshold=0.5, color_map=None)
# cv2.imshow('frame', frame)
