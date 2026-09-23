# --- Stage 1: build the React frontend ---
FROM node:20-slim AS frontend-builder
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- Stage 2: Python application ---
FROM python:3.11-slim AS app

WORKDIR /app

# System deps needed by faiss-cpu and pyserini (Java not included by default)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so editing source does not reinstall them.
COPY requirements.txt ./

# CPU-only torch before requirements.txt. The default linux torch wheel drags in
# 33 GPU packages -- nvidia-cublas, nvidia-cudnn-cu13, nvidia-nccl-cu13,
# cuda-toolkit, triton and the rest -- and nothing this image runs can use them:
# every serving device default is "cpu" (document_index/embedding.py,
# document_index/retrieval.py, memory/service.py), the one cuda selection is the
# offline index-build CLI which degrades through torch.cuda.is_available(), and
# the compose stack maps no GPU device. Installing the CPU wheel first leaves
# torch>=2.3.0 already satisfied, so the line below does not reach for it.
# The CI job asserts no nvidia/cuda/triton package survives in the image.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Bring in the pre-built frontend bundle. src/internal/servers/web/app.py
# resolves it as parents[4]/"web"/"dist", which is /app/web/dist here.
COPY --from=frontend-builder /app/web/dist ./web/dist

# The package is installed only now, after the source exists. `pip install -e .`
# before COPY discovers zero packages under [tool.setuptools.packages.find], so
# setuptools writes an editable finder with an empty MAPPING: `import src` then
# resolves only because WORKDIR puts the cwd on sys.path, and any process with a
# different cwd raises ModuleNotFoundError. --no-deps because every runtime
# dependency, PyJWT included, is already pinned in requirements.txt.
RUN pip install --no-cache-dir --no-deps -e .

# Run as a non-root user. The order matters twice over: the pip installs above
# need root to write site-packages, and /data must be created *in the image*
# owned by this user, because docker seeds a fresh named volume with the
# ownership of the directory it mounts over. AgenticSearchStore opens
# AGENTIC_SEARCH_WEB_DB_PATH during create_web_app(), so a /data the app cannot
# write crashes startup rather than failing later -- which is why the compose
# healthcheck catches it.
#
# An EXISTING volume keeps whatever ownership it already had. A stack that ran
# as root before this change needs `docker compose down -v`, or a one-off
# `docker run --rm -v docker_app_data:/data alpine chown -R 10001:10001 /data`.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown app:app /data

# HOME must be writable: sentence-transformers and transformers cache model
# downloads under it, which the flag-gated memory encoder does when
# AGENTIC_SEARCH_MEMORY_SEMANTIC is set.
ENV HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface

USER app

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

# Deliberately one worker. app.state holds per-process objects that later
# requests depend on finding again: the ToolApprovalBroker (a client polls for a
# decision and must reach the worker that created it), the RequestCaptureStore
# the Dev Console reads back, the loaded search-agent model, and the memory
# encoder. Scaling this line out makes tool approvals fail intermittently, loses
# Dev Console captures, and loads the model once per worker. Postgres and Redis
# are already in the compose stack; moving the broker and the capture store
# there is the prerequisite for running more than one worker.
CMD ["uvicorn", "src.internal.servers.web.app:app", "--host", "0.0.0.0", "--port", "7860"]
