FROM python:3.8-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装MinerU
RUN pip install mineru

# 复制项目文件
COPY . .

# 创建模型缓存目录
RUN mkdir -p /root/.cache/modelscope/hub

# 暴露端口
EXPOSE 5002

# 设置环境变量
ENV BASE_URL=http://localhost:5002
ENV MINERU_MODEL_SOURCE=modelscope

# 启动服务
CMD ["python", "service.py"]