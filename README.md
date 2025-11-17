# MinerU PDF解析服务

基于FastAPI的PDF解析服务，使用MinerU工具解析PDF文件并返回Markdown格式内容及图片。

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.68%2B-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## 功能特性

- 上传PDF文件并解析为Markdown内容
- 提供图片的HTTP访问链接
- 支持HTML格式输出
- 自动清理临时文件
- 提供图片URL转Base64数据URL格式的专用接口
- 高性能异步处理

## 目录结构

```
.
├── service.py          # 主服务文件
├── requirements.txt    # 依赖包列表
├── README.md           # 项目说明文档
├── LICENSE             # 许可证文件
├── Dockerfile          # Docker配置文件
├── .gitignore          # Git忽略文件
├── test_service.py     # 服务测试脚本
└── temp_output/        # 临时文件存储目录
```

## 快速开始

### 环境要求

- Python 3.8+
- pip包管理器
- MinerU工具已安装并可访问（[参考mineru部署教程](https://www.51cto.com/article/821228.html)）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行服务

```bash
python service.py
```

服务将启动在 `http://0.0.0.0:5002`

### 测试服务

使用提供的测试脚本验证服务是否正常运行：

```bash
python test_service.py
```

## API接口

### 1. 上传文件解析PDF (`/upload`)

**请求方法**: POST

**表单参数**:
- `file`: 上传的PDF文件
- `lang`: 语言设置，默认为"ch"
- `backend`: 解析后端，默认为"pipeline"
- `method`: 解析方法，默认为"auto"
- `output_format`: 输出格式，可选"md"或"html"，默认为"md"

**请求示例**:

```bash
curl -X POST "http://localhost:5002/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example.pdf" \
  -F "lang=ch" \
  -F "backend=pipeline" \
  -F "method=auto" \
  -F "output_format=md"
```

**响应示例**:
```json
{
  "content": "解析后的Markdown内容",
  "images": [
    "http://localhost:5002/temp_output/task_id/name/auto/images/image1.png"
  ]
}
```

### 2. 图片URL转Base64数据URL格式 (`/image-to-base64`)

**请求方法**: POST

**请求体**:
```json
{
  "image_url": "http://example.com/image.png"
}
```

**请求示例**:

```bash
curl -X POST "http://localhost:5002/image-to-base64" \
  -H "accept: text/plain" \
  -H "Content-Type: application/json" \
  -d '{"image_url":"http://localhost:5002/temp_output/task_id/name/auto/images/image1.png"}'
```

**响应示例**:
```
data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==
```

## 部署配置

### 环境变量

- `BASE_URL`: 服务的基础URL，用于生成图片访问链接，默认为"http://localhost:5002"
- `MINERU_MODEL_SOURCE`: MinerU模型源，默认为"modelscope"

### Docker部署

创建Dockerfile:

```dockerfile
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
```

构建和运行:

```bash
docker build -t mineru-service .
docker run -p 5002:5002 -e BASE_URL=http://your-domain.com:5002 mineru-service
```

## 依赖项

- fastapi
- uvicorn[standard]
- markdown
- pydantic
- python-multipart
- requests

## 代码结构说明

项目代码按照功能模块进行了组织，便于理解和维护：

- **全局配置和常量**: 包含日志配置、路径设置等
- **应用初始化**: FastAPI应用实例和相关配置
- **数据模型**: Pydantic数据模型定义
- **工具函数**: 辅助函数，如MIME类型识别等
- **核心业务逻辑**: MinerU调用等核心功能
- **异步任务处理**: 异步执行CPU密集型任务
- **后台任务**: 定时清理等后台任务
- **API接口**: 所有HTTP接口实现
- **应用启动**: 服务启动配置

这种结构化的组织方式使得代码更易于维护和扩展。

## 性能优化

- 使用异步处理提高并发性能
- 实现LRU缓存避免重复计算
- 优化文件清理机制减少资源占用
- 使用连接池复用HTTP连接

## 许可证

本项目采用MIT许可证，详情请见[LICENSE](LICENSE)文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。