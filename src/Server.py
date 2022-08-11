# 推理类

import os, io, uuid
from PIL import Image
import flask, json
import cv2
import numpy as np
import paddlex as pdx
import urllib.request as httpclient
import tempfile
from flask import request
from flask import send_file
from flask import make_response

# 工作空间
_workspace = os.path.join(os.path.dirname(__file__), '..')

# 初始化Server
server = flask.Flask(__name__)
# 预测模型加载
# model = pdx.load_model(os.path.join(_workspace, 'best_model'))
predictor = pdx.deploy.Predictor(os.path.join(_workspace, 'inference_model'))


# server
class Server():

    def __init__(self):
        print('Server init ...')

    def start(self, port):  # 私有方法
        print('server start ...')
        server.run(debug=False, port=port, host='0.0.0.0')  # 指定端口,host,0.0.0.0代表不管几个网卡，任何ip都可访问


@server.route('/image', methods=['get', 'post'])
def image():
    # 图片
    url = request.values.get('url')
    file = flask.request.files.get('file')

    # 门槛
    threshold = float(request.values.get('threshold', 0.5))
    # 是否可视化
    visualize = True if request.values.get('visualize', 'true') == 'true' else False

    imageBuffer = None
    buffer = None
    # 判断url或者body优先
    if url is not None:
        response = httpclient.urlopen(url)
        imageBuffer = bytearray(response.read())
        buffer = cv2.imdecode(np.array(imageBuffer, dtype=np.uint8), -1)
    elif file is not None:
        imageBuffer = bytearray(file.stream.read())
        buffer = cv2.imdecode(np.array(imageBuffer, dtype=np.uint8), -1)
    if buffer is None:
        return json.dumps({'code': 500, 'message': 'url or file not null', 'ret': None}, ensure_ascii=False)

    # 构建临时目录
    tmpOutput = tempfile.TemporaryDirectory(str(uuid.uuid1()))

    # 预测
    result = predictor.predict(buffer)
    resultImage = None

    # 可视化保存
    if visualize == True:
        tmpPath = tmpOutput.name
        pdx.det.visualize(buffer, result, threshold=threshold, save_dir=tmpPath)
        for file in os.listdir(tmpPath):
            break

        # frame = cv2.imread(fileName)
        # resultImage = io.BytesIO(cv2.imencode('.jpg', frame)[1])

        fileName = os.path.join(tmpPath, file)
        file = open(fileName, 'br')
        resultImage = io.BytesIO(file.read())
        file.close()
    else:
        resultImage = io.BytesIO()

    # 构建响应
    response = make_response(send_file(
        resultImage,
        mimetype='image/jpeg'
    ))
    # 预测结果
    response.headers['predict'] = json.dumps(result, ensure_ascii=False)
    return response

