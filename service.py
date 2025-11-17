import os
import uuid
import shutil
import time
import json
import asyncio
import logging
import base64
import requests
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import markdown
import uvicorn
# 添加 subprocess 模块用于更高效地执行命令
import subprocess
# 添加 functools 和 lru_cache 用于缓存
from functools import lru_cache
# 添加线程池执行器
from concurrent.futures import ThreadPoolExecutor
# 添加请求重试相关模块
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse

# ======================
# 全局配置和常量
# ======================

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mineru")

# 临时输出目录
TEMP_BASE_DIR = Path("./temp_output")
TEMP_BASE_DIR.mkdir(exist_ok=True)

# 获取服务基础URL的配置，如果没有设置则使用默认值
BASE_URL = os.environ.get("BASE_URL", "http://localhost:5002")

# 模型是否已检查的标志
MODEL_CHECKED = False

# ======================
# 应用初始化
# ======================

# 创建 FastAPI 应用
app = FastAPI()

# 创建 requests session 以复用连接
requests_session = requests.Session()
# 配置重试策略
retry_strategy = Retry(
    total=3,  # 总重试次数
    backoff_factor=1,  # 重试间隔时间的倍数
    status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码
)
adapter = HTTPAdapter(max_retries=retry_strategy)
requests_session.mount("http://", adapter)
requests_session.mount("https://", adapter)

# 创建线程池执行器以更好地管理异步任务
thread_pool_executor = ThreadPoolExecutor(max_workers=4)

# 挂载静态文件目录，用于访问图片
app.mount("/temp_output", StaticFiles(directory="temp_output"), name="temp_output")

# ======================
# 数据模型
# ======================

# 图片URL转base64的请求模型
class ImageUrlRequest(BaseModel):
    image_url: str

# ======================
# 工具函数
# ======================

def get_mime_type_from_extension(extension: str) -> str:
    """根据文件扩展名确定MIME类型"""
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif'
    }
    return mime_types.get(extension.lower(), 'image/png')

# 缓存图片信息以提高性能
@lru_cache(maxsize=128)
def get_cached_image_urls(task_id: str, name: str) -> list:
    """缓存图片URL列表以避免重复计算"""
    image_urls = []
    images_dir = TEMP_BASE_DIR / task_id / name / "auto" / "images"
    if images_dir.exists():
        for img_file in images_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                # 生成可通过HTTP访问的图片URL
                image_url = f"{BASE_URL}/temp_output/{task_id}/{name}/auto/images/{img_file.name}"
                image_urls.append(image_url)
    return image_urls

# 清理不必要的文件，只保留Markdown和图片
def cleanup_unnecessary_files(task_dir: Path, name: str):
    """清理不必要的文件，只保留Markdown和图片"""
    # 保留的文件和目录
    auto_dir = task_dir / name / "auto"
    
    if auto_dir.exists():
        # 遍历auto目录中的所有项目
        for item in auto_dir.iterdir():
            if item.is_file():
                # 只保留Markdown文件
                if not item.name.endswith('.md'):
                    item.unlink()
            elif item.is_dir():
                # 只保留images目录
                if item.name == "images":
                    # 清理images目录中非图片文件
                    for img_file in item.iterdir():
                        if img_file.is_file() and img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.gif']:
                            img_file.unlink()
                else:
                    # 删除其他目录
                    shutil.rmtree(item, ignore_errors=True)
    
    # 删除根目录下除了原始PDF之外的其他文件
    for item in task_dir.iterdir():
        if item.is_file() and item.name != f"{name}.pdf":
            item.unlink()

# ======================
# 核心业务逻辑
# ======================

# 调用 mineru 命令行接口，使用 subprocess 替代 os.system 提高性能
def run_mineru_cli(input_path: str, output_path: str, lang: str = "ch", backend: str = "pipeline",
                   method: str = "auto"):
    global MODEL_CHECKED
    
    # 设置模型源为modelscope
    os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
    
    # 只在第一次运行时检查模型，避免重复检查
    if not MODEL_CHECKED:
        logger.info("首次运行，MinerU将检查模型")
        MODEL_CHECKED = True
    else:
        logger.info("模型已检查过，跳过重复检查")
    
    cmd = ["mineru", "-p", input_path, "-o", output_path, "--source", "local"]
    logger.info(f"运行命令: {' '.join(cmd)}")
    
    # 使用 subprocess 替代 os.system，提供更好的错误处理和性能
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分钟超时
        if result.returncode != 0:
            raise RuntimeError(f"mineru 命令执行失败，退出码 {result.returncode}: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("mineru 命令执行超时")
    except Exception as e:
        raise RuntimeError(f"执行 mineru 命令时发生错误: {str(e)}")

# ======================
# 异步任务处理
# ======================

# 异步执行 mineru 命令以提高并发性能
async def async_run_mineru_cli(input_path: str, output_path: str, lang: str = "ch", backend: str = "pipeline",
                               method: str = "auto"):
    """异步执行 mineru 命令以提高并发性能"""
    loop = asyncio.get_event_loop()
    # 使用线程池执行器运行 CPU 密集型任务
    return await loop.run_in_executor(thread_pool_executor, run_mineru_cli, input_path, output_path, lang, backend, method)

# 异步清理不必要的文件任务
async def async_cleanup_unnecessary_files(task_dir: Path, name: str):
    """异步清理不必要的文件，不阻塞主请求处理"""
    loop = asyncio.get_event_loop()
    # 在线程池中运行文件清理任务
    await loop.run_in_executor(thread_pool_executor, cleanup_unnecessary_files, task_dir, name)

# ======================
# 后台任务
# ======================

# 启动清理任务（清除超过 30min 的目录）
async def clean_temp_dirs():
    while True:
        try:
            for dir in TEMP_BASE_DIR.iterdir():
                if dir.is_dir() and (time.time() - dir.stat().st_mtime) > 1800:
                    shutil.rmtree(dir, ignore_errors=True)
                    logger.info(f"清理目录: {dir}")
        except Exception as e:
            logger.exception(f"清理任务异常: {e}")
        # 减少清理频率以降低资源消耗
        await asyncio.sleep(3600)  # 每小时清理一次而不是每30分钟

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(clean_temp_dirs())

@app.on_event("shutdown")
async def shutdown_event():
    # 关闭线程池执行器
    thread_pool_executor.shutdown(wait=True)

# ======================
# API接口
# ======================

@app.post("/upload")
async def parse_from_upload(
        file: UploadFile = File(...),
        lang: str = Form("ch"),
        backend: str = Form("pipeline"),
        method: str = Form("auto"),
        output_format: str = Form("md")
):
    start_time = time.time()
    content = await file.read()
    # 处理文件名，确保不包含非法字符
    raw_filename = file.filename or f"uploaded_file_{uuid.uuid4().hex[:8]}"
    clean_filename = "".join(c for c in raw_filename if c.isalnum() or c in (' ', '.', '_', '-')).strip()
    # 确保文件有.pdf扩展名
    if not clean_filename.endswith('.pdf'):
        clean_filename += '.pdf'
    filename = clean_filename
    name = Path(filename).stem
    task_id = uuid.uuid4().hex[:8]
    task_dir = TEMP_BASE_DIR / task_id
    task_dir.mkdir(exist_ok=True)

    input_path = task_dir / filename

    with open(input_path, "wb") as f:
        f.write(content)

    # 使用异步方式执行 mineru 命令
    await async_run_mineru_cli(str(input_path), str(task_dir), lang=lang, backend=backend, method=method)

    output_md_path = task_dir / name / "auto"
    output_md_path = output_md_path / f"{name}.md"
    
    # 检查Markdown文件是否存在
    if not output_md_path.exists():
        # 尝试另一种可能的路径
        output_md_path = task_dir / f"{name}.md"
        
    if not output_md_path.exists():
        return JSONResponse(content={"error": "未找到解析后的Markdown文件"}, status_code=500)
    
    # 读取Markdown内容
    md_content = output_md_path.read_text(encoding="utf-8")
    
    # 使用缓存获取图片URL列表
    image_urls = get_cached_image_urls(task_id, name)
    
    # 异步启动清理任务，不阻塞主请求处理
    asyncio.create_task(async_cleanup_unnecessary_files(task_dir, name))
    
    # 根据输出格式返回结果，延迟处理HTML转换
    if output_format == "html":
        # 只在需要时才进行Markdown到HTML的转换
        html_content = markdown.markdown(md_content, extensions=['extra', 'tables', 'nl2br'])
        html_content = html_content.replace("\n", "")  # 可选：去除残余换行符
        result = {
            "content": html_content,
            "images": image_urls
        }
    else:
        result = {
            "content": md_content,
            "images": image_urls
        }
    
    # 在控制台打印响应的图片信息
    logger.info(f"返回的图片数量: {len(image_urls)}")
    for i, image_url in enumerate(image_urls):
        logger.info(f"图片 {i+1}: {image_url}")
        
    return JSONResponse(content=result)

@app.post("/image-to-base64")
async def convert_image_to_base64(req: ImageUrlRequest):
    """将图片URL转换为data:image/TYPE;base64,YOUR-BASE64-CONTENT格式"""
    try:
        # 验证URL格式
        parsed_url = urlparse(req.image_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.error(f"无效的图片URL格式: {req.image_url}")
            return Response(content="无效的图片URL格式", status_code=400, media_type="text/plain")
        
        # 记录请求信息
        logger.info(f"开始转换图片: {req.image_url}")
        
        # 检查是否是本地URL
        is_local_url = parsed_url.hostname in ['localhost', '127.0.0.1'] or parsed_url.hostname is None
        image_data = None
        
        if is_local_url:
            logger.info(f"检测到本地URL: {req.image_url}")
            
            # 对于本地URL，直接读取本地文件而不是通过HTTP请求
            # 从URL中提取文件路径
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) >= 4 and path_parts[0] == 'temp_output':
                # 提取task_id和文件名
                task_id = path_parts[1]
                file_name = path_parts[-1]
                
                # 构建本地文件路径
                local_file_path = TEMP_BASE_DIR / task_id / path_parts[2] / "auto" / "images" / file_name
                logger.info(f"检查本地文件路径: {local_file_path}")
                
                # 检查文件是否存在
                if local_file_path.exists():
                    # 直接读取本地文件
                    image_data = local_file_path.read_bytes()
                    logger.info(f"成功读取本地文件: {local_file_path}, 大小: {len(image_data)} 字节")
                else:
                    logger.error(f"本地文件不存在: {local_file_path}")
                    return Response(content=f"本地文件不存在: {file_name}", status_code=400, media_type="text/plain")
        
        # 如果不是本地URL或者本地文件读取失败，则通过HTTP请求下载
        if image_data is None:
            # 使用 session 下载图片以复用连接，并设置合理的超时
            response = requests_session.get(req.image_url, timeout=(10, 30))  # 连接超时10秒，读取超时30秒
            response.raise_for_status()
            
            # 检查响应内容类型
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"URL可能不是图片: {req.image_url}, Content-Type: {content_type}")
            
            # 获取图片内容
            image_data = response.content
            content_length = len(image_data)
            logger.info(f"成功下载图片: {req.image_url}, 大小: {content_length} 字节")
        
        # 确定图片的MIME类型
        file_extension = Path(parsed_url.path).suffix if Path(parsed_url.path).suffix else '.png'
        mime_type = get_mime_type_from_extension(file_extension)
        logger.info(f"图片MIME类型: {mime_type}, 文件扩展名: {file_extension}")
        
        # 转换为base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        logger.info(f"Base64编码完成，长度: {len(image_base64)} 字符")
        
        # 构造符合要求的数据URL格式
        data_url = f"data:{mime_type};base64,{image_base64}"
        
        # 在控制台打印成功信息
        logger.info(f"成功转换图片: {req.image_url}")
        
        # 直接返回字符串而不是JSON
        return Response(content=data_url, media_type="text/plain")
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"连接图片服务器失败: {req.image_url}, 错误: {str(e)}")
        return Response(content=f"连接图片服务器失败: 请检查URL是否正确且可访问", status_code=400, media_type="text/plain")
    except requests.exceptions.Timeout as e:
        logger.error(f"下载图片超时: {req.image_url}, 错误: {str(e)}")
        return Response(content=f"下载图片超时: 请稍后重试或检查网络连接", status_code=400, media_type="text/plain")
    except requests.exceptions.RetryError as e:
        logger.error(f"重试次数耗尽，无法下载图片: {req.image_url}, 错误: {str(e)}")
        return Response(content=f"重试次数耗尽，无法下载图片: 请稍后重试", status_code=400, media_type="text/plain")
    except requests.RequestException as e:
        logger.error(f"下载图片失败: {req.image_url}, 错误: {str(e)}")
        return Response(content=f"下载图片失败: {str(e)}", status_code=400, media_type="text/plain")
    except Exception as e:
        logger.error(f"处理图片时出错: {req.image_url}, 错误: {str(e)}")
        return Response(content=f"处理图片时出错: {str(e)}", status_code=500, media_type="text/plain")

# ======================
# 应用启动
# ======================

# 启动服务
if __name__ == '__main__':
    uvicorn.run("__main__:app", host="0.0.0.0", port=5002, reload=False)