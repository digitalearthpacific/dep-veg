# syntax=docker/dockerfile:1
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.7.3

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV USE_PYGEOS=0
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install uv (The pip replacement)
COPY --from=ghcr.io/astral-sh/uv:latest  /uv /uvx /bin/

# Install System Deps (Cached)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    git \
    libpq-dev \
    build-essential \
    libcairo2-dev \
    pkg-config \
    libgirepository1.0-dev \
    gobject-introspection \
    libglib2.0-dev \
    libdbus-1-dev \
    rustc \
    cargo

# 3. Create Virtual Environment
# We make it here so subsequent commands run inside it automatically
# thanks to the PATH variable set above.
RUN uv venv /opt/venv

# 4. Install Heavy Deps separately (Torch)
# We do this BEFORE requirements.txt.
# If you edit requirements.txt, this massive download is skipped (cached layer).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 5. Install Project Requirements
# Using --mount=type=cache allows uv to reuse downloads across builds
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    uv pip install -r /tmp/requirements.txt

COPY . /code
WORKDIR /code