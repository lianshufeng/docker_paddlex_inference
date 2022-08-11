FROM paddlepaddle/paddle:2.3.1
MAINTAINER lianshufeng <251708339@qq.com>

# 添加 依赖
Add ./ /infer

# 安装依赖
RUN pip install -r /infer/requirements.txt

# 运行服务
CMD ["python","src/ApplicationMain.py"]