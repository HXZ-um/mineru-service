import os
import uuid
import shutil
import time
import asyncio
import logging
import base64
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import markdown
import uvicorn

# ======================
# 全局配置
# ======================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mineru_server")

TEMP_BASE_DIR = Path("./temp_output").resolve()
TEMP_BASE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = os.environ.get("BASE_URL", "http://localhost:5002").rstrip("/")

# MinerU 配置文件路径（可选，强烈建议放英文目录）
MINERU_CONFIG_JSON = os.environ.get("MINERU_TOOLS_CONFIG_JSON")

# 默认模型源：local / modelscope / huggingface
DEFAULT_MODEL_SOURCE = os.environ.get("MINERU_MODEL_SOURCE", "local")

# 并发限制：同时最多跑几个 mineru 子进程
MAX_MINERU_CONCURRENCY = int(os.environ.get("MAX_MINERU_CONCURRENCY", "2"))
mineru_semaphore = asyncio.Semaphore(MAX_MINERU_CONCURRENCY)

# 线程池：只给阻塞型任务用（subprocess、文件清理）
thread_pool_executor = ThreadPoolExecutor(max_workers=4)

# requests session + retry
requests_session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
requests_session.mount("http://", adapter)
requests_session.mount("https://", adapter)

# ======================
# FastAPI 初始化
# ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(clean_temp_dirs())
    yield
    task.cancel()
    thread_pool_executor.shutdown(wait=True)
app = FastAPI(lifespan=lifespan)

# 静态文件挂载：确保用绝对路径，不受 cwd 影响
app.mount("/temp_output", StaticFiles(directory=str(TEMP_BASE_DIR)), name="temp_output")

# ======================
# 数据模型
# ======================

class ImageUrlRequest(BaseModel):
    image_url: str

# ======================
# 工具函数
# ======================

def get_mime_type_from_extension(extension: str) -> str:
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
    }
    return mime_types.get(extension.lower(), "image/png")


def list_image_urls(task_id: str, name: str) -> List[str]:
    """不缓存，避免返回旧图。一次任务里 I/O 很轻。"""
    image_urls = []
    images_dir = TEMP_BASE_DIR / task_id / name / "auto" / "images"
    if images_dir.exists():
        for img_file in images_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif"]:
                image_urls.append(
                    f"{BASE_URL}/temp_output/{task_id}/{name}/auto/images/{img_file.name}"
                )
    return image_urls


def cleanup_unnecessary_files(task_dir: Path, name: str):
    """
    清理不需要文件：
    - 保留 auto 下的 .md 和 images
    - 根目录仅保留原始 pdf
    """
    auto_dir = task_dir / name / "auto"

    if auto_dir.exists():
        for item in auto_dir.iterdir():
            if item.is_file():
                if item.suffix.lower() != ".md":
                    item.unlink(missing_ok=True)
            elif item.is_dir():
                if item.name == "images":
                    for f in item.iterdir():
                        if f.is_file() and f.suffix.lower() not in [".png", ".jpg", ".jpeg", ".gif"]:
                            f.unlink(missing_ok=True)
                else:
                    shutil.rmtree(item, ignore_errors=True)

    for item in task_dir.iterdir():
        if item.is_file() and item.name != f"{name}.pdf":
            item.unlink(missing_ok=True)


def find_output_markdown(task_dir: Path, name: str) -> Optional[Path]:
    """
    更鲁棒地找 md:
    1. task_dir/name/auto/*.md
    2. task_dir/name/*.md
    3. task_dir/*.md
    """
    candidates = []

    auto_dir = task_dir / name / "auto"
    if auto_dir.exists():
        candidates.extend(sorted(auto_dir.glob("*.md")))

    name_dir = task_dir / name
    if name_dir.exists():
        candidates.extend(sorted(name_dir.glob("*.md")))

    candidates.extend(sorted(task_dir.glob("*.md")))

    return candidates[0] if candidates else None


def build_mineru_cmd(input_path: str, output_path: str, lang: str, backend: str, method: str, source: str):
    """
    根据参数拼 mineru 命令。
    注意：只用一个 source，不再 env/cli 冲突。
    """
    cmd = [
        "mineru",
        "-p", input_path,
        "-o", output_path,
        "--lang", lang,
        "--backend", backend,
        "--method", method,
        "--source", source,
    ]
    return cmd


def run_mineru_cli(input_path: str, task_dir: str, lang="ch", backend="pipeline", method="auto", source="local"):
    """
    运行 mineru 子进程（同步阻塞）。
    - 明确 source
    - 兼容 Windows 编码：utf-8 + replace
    """
    env = os.environ.copy()
    env["MINERU_MODEL_SOURCE"] = source
    if MINERU_CONFIG_JSON:
        env["MINERU_TOOLS_CONFIG_JSON"] = MINERU_CONFIG_JSON

    cmd = build_mineru_cmd(input_path, task_dir, lang, backend, method, source)
    logger.info(f"MinerU CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("mineru 命令执行超时（>300s）")
    except FileNotFoundError:
        raise RuntimeError("找不到 mineru 命令，请确认已安装并可在 PATH 里调用")
    except Exception as e:
        raise RuntimeError(f"执行 mineru 时发生异常: {e}")

    if result.returncode != 0:
        logger.error("MinerU STDERR:\n" + (result.stderr or ""))
        logger.error("MinerU STDOUT:\n" + (result.stdout or ""))
        raise RuntimeError(f"mineru 执行失败，exit={result.returncode}")

    return True


async def async_run_mineru_cli(input_path: str, task_dir: str, lang="ch", backend="pipeline", method="auto", source="local"):
    """
    异步包装：
    - 用 semaphore 控制并发 mineru 数量
    - 用线程池执行阻塞 subprocess
    """
    async with mineru_semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            thread_pool_executor,
            run_mineru_cli,
            input_path, task_dir, lang, backend, method, source
        )


async def async_cleanup(task_dir: Path, name: str):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(thread_pool_executor, cleanup_unnecessary_files, task_dir, name)


async def clean_temp_dirs():
    """
    清理超过 30min 的任务目录
    """
    while True:
        try:
            now = time.time()
            for d in TEMP_BASE_DIR.iterdir():
                if d.is_dir() and (now - d.stat().st_mtime) > 1800:
                    shutil.rmtree(d, ignore_errors=True)
                    logger.info(f"清理目录: {d}")
        except Exception as e:
            logger.exception(f"清理任务异常: {e}")

        await asyncio.sleep(3600)







# ======================
# API
# ======================

@app.post("/upload")
async def parse_from_upload(
    file: UploadFile = File(...),
    lang: str = Form("ch"),
    backend: str = Form("pipeline"),
    method: str = Form("auto"),
    output_format: str = Form("md"),
    model_source: str = Form(DEFAULT_MODEL_SOURCE),
):
    """
    上传 PDF -> MinerU 解析 -> 返回 md/html + 图片 URL
    model_source: local / modelscope / huggingface
    """
    start_time = time.time()

    content = await file.read()

    raw_filename = file.filename or f"uploaded_{uuid.uuid4().hex[:8]}.pdf"
    clean_filename = "".join(
        c for c in raw_filename if c.isalnum() or c in (" ", ".", "_", "-")
    ).strip()

    if not clean_filename.lower().endswith(".pdf"):
        clean_filename += ".pdf"

    filename = clean_filename
    name = Path(filename).stem
    task_id = uuid.uuid4().hex[:8]
    task_dir = TEMP_BASE_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    input_path = task_dir / filename
    input_path.write_bytes(content)

    try:
        await async_run_mineru_cli(
            input_path=str(input_path),
            task_dir=str(task_dir),
            lang=lang,
            backend=backend,
            method=method,
            source=model_source,
        )
    except Exception as e:
        return JSONResponse(
            content={
                "error": str(e),
                "task_id": task_id,
            },
            status_code=500,
        )

    md_path = find_output_markdown(task_dir, name)
    if not md_path or not md_path.exists():
        return JSONResponse(
            content={
                "error": "未找到解析后的 Markdown 文件",
                "task_id": task_id,
            },
            status_code=500,
        )

    # 读 md（显式 utf-8）
    md_content = md_path.read_text(encoding="utf-8", errors="replace")

    image_urls = list_image_urls(task_id, name)

    # 异步清理（不阻塞）
    asyncio.create_task(async_cleanup(task_dir, name))

    if output_format.lower() == "html":
        html_content = markdown.markdown(md_content, extensions=["extra", "tables", "nl2br"])
        html_content = html_content.replace("\n", "")
        result = {"content": html_content, "images": image_urls}
    else:
        result = {"content": md_content, "images": image_urls}

    logger.info(
        f"任务完成 task_id={task_id}, md={md_path.name}, images={len(image_urls)}, "
        f"elapsed={time.time()-start_time:.2f}s"
    )
    return JSONResponse(content=result)


@app.post("/image-to-base64")
async def convert_image_to_base64(req: ImageUrlRequest):
    try:
        parsed_url = urlparse(req.image_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return Response("无效的图片URL格式", status_code=400, media_type="text/plain")

        is_local_url = parsed_url.hostname in ["localhost", "127.0.0.1"] or parsed_url.hostname is None

        image_data = None
        if is_local_url:
            # 只要是 /temp_output/... 就按本地读
            path_parts = parsed_url.path.strip("/").split("/")
            # temp_output/{task_id}/{name}/auto/images/{file}
            if len(path_parts) >= 6 and path_parts[0] == "temp_output":
                task_id = path_parts[1]
                name = path_parts[2]
                file_name = path_parts[-1]
                local_file_path = TEMP_BASE_DIR / task_id / name / "auto" / "images" / file_name
                if local_file_path.exists():
                    image_data = local_file_path.read_bytes()
                else:
                    return Response(f"本地文件不存在: {file_name}", status_code=400, media_type="text/plain")

        if image_data is None:
            resp = requests_session.get(req.image_url, timeout=(10, 30))
            resp.raise_for_status()
            image_data = resp.content

        ext = Path(parsed_url.path).suffix or ".png"
        mime_type = get_mime_type_from_extension(ext)

        image_base64 = base64.b64encode(image_data).decode("utf-8")
        data_url = f"data:{mime_type};base64,{image_base64}"
        return Response(content=data_url, media_type="text/plain")

    except requests.exceptions.RequestException as e:
        return Response(f"下载图片失败: {e}", status_code=400, media_type="text/plain")
    except Exception as e:
        return Response(f"处理图片出错: {e}", status_code=500, media_type="text/plain")


# ======================
# 启动
# ======================

if __name__ == "__main__":
    uvicorn.run("service:app", host="0.0.0.0", port=5002, reload=False)

