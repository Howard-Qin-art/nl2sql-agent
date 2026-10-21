# NL2SQL Agent 镜像：API 服务（LLM 走外部 Ollama 或云端 API）
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 依赖单独一层，改代码不会触发重装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY eval ./eval
COPY ui ./ui
COPY scripts ./scripts
COPY tests ./tests

# 构建镜像时就把示例库生成好，容器起来即可查询
RUN python -m app.db.seed

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"

# 只读挂载数据库时用：-v $PWD/data:/app/data:ro
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
