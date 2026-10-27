# NL2SQL Agent 镜像：API 服务（LLM 走外部 Ollama 或云端 API）
#
# 两个 ARG 是为了「国内网络」也能构建：
#   docker build -t nl2sql-agent:dev \
#     --build-arg BASE_IMAGE=docker.m.daocloud.io/library/python:3.10-slim \
#     --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
# 不传就是官方源，海外环境不用改。
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 依赖单独一层 + BuildKit 缓存挂载：requirements 有变动时只重装变化的包，
# 不会每次都把 langgraph/pandas/streamlit 全下一遍。
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i ${PIP_INDEX_URL} -r requirements.txt

COPY app ./app
COPY eval ./eval
COPY ui ./ui
COPY scripts ./scripts
COPY tests ./tests
COPY mcp_server.py .

# 构建镜像时就把示例库生成好，容器起来即可查询
RUN python -m app.db.seed

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"

# 只读挂载数据库时用：-v $PWD/data:/app/data:ro
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
